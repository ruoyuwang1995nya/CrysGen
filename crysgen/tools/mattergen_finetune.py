"""Local finetuning entry point for MatterGen.

This module is imported by remote submission code only to access the
FinetuneOptions dataclass. To avoid requiring heavy training dependencies
for submission, most deep imports are deferred until `run_finetune`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from .base_tools import Tools
import logging

try:  # minimal torch usage for accelerator resolution; fallback to CPU if missing
    import torch  # type: ignore
    print("OK")
except ModuleNotFoundError:  # pragma: no cover
    class _TorchStub:  # noqa: D401
        @staticmethod
        def cuda():  # mimic torch.cuda
            class _C:
                @staticmethod
                def is_available():
                    return False
            return _C()

        @staticmethod
        def set_float32_matmul_precision(*_args, **_kwargs):
            return None

    torch = _TorchStub()  # type: ignore

# Heavy imports (Hydra, Lightning, OmegaConf, mattergen) are deferred to run_finetune/compose_config.


@dataclass(slots=True)
class FinetuneOptions:
    #data_root: Path
    data_train: Path
    data_val: Path
    data_test: Path | None
    model_dir: Path
    output_dir: Path
    max_epochs: int
    train_batch_size: int
    val_batch_size: int
    num_workers: int
    devices: int
    accelerator: str
    strategy: str
    precision: str
    resume: Path | None
    use_wandb: bool
    properties: list | None 
    env: dict
    optional_args: list

    @property
    def effective_accelerator(self) -> str:
        if self.accelerator != "auto":
            return self.accelerator
        return "gpu" if torch.cuda.is_available() else "cpu"

def _build_overrides(options: FinetuneOptions) -> List[str]:
    model_dir = options.model_dir.resolve()

    overrides = [
        f"data_module.train_dataset.cache_path={options.data_train.resolve().as_posix()}",
        f"data_module.val_dataset.cache_path={options.data_val.resolve().as_posix()}",
        f"data_module.test_dataset.cache_path={options.data_test.resolve().as_posix()}" if options.data_test else "~data_module.test_dataset",
        f"adapter.model_path={model_dir.as_posix()}",
        "adapter.pretrained_name=null",
        f"trainer.max_epochs={options.max_epochs}",
        f"data_module.max_epochs={options.max_epochs}",
        f"trainer.devices={options.devices}",
        f"trainer.accelerator={options.effective_accelerator}",
        f"trainer.precision={options.precision}",
        #f"trainer.strategy={options.strategy}",
        f"data_module.batch_size.train={options.train_batch_size}",
        f"data_module.batch_size.val={options.val_batch_size}",
        f"data_module.batch_size.test={options.val_batch_size}",
        f"data_module.num_workers.train={options.num_workers}",
        f"data_module.num_workers.val={options.num_workers}",
        f"data_module.num_workers.test={max(0, options.num_workers // 2)}",
    ]
    #if options.properties:
    if len(options.properties) > 0:
        prop_ls=[]
        for prop in options.properties:
            overrides.append(f'+lightning_module/diffusion_module/model/property_embeddings@adapter.adapter.property_embeddings_adapt.{prop}={prop}')
            prop_ls.append(prop)
        overrides.append(f"data_module.properties={prop_ls}")
    if not options.use_wandb:
        overrides.append("~trainer.logger")
    for arg in options.optional_args:
        overrides.append(arg)
    return overrides


def compose_config(overrides: List[str]):
    try:
        from hydra import compose, initialize_config_dir  # type: ignore
        from hydra.core.global_hydra import GlobalHydra  # type: ignore
        from mattergen.common.utils.globals import MODELS_PROJECT_ROOT  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "Hydra 未安装。请在需要本地训练/微调时先执行: pip install hydra-core omegaconf"
        ) from e
    config_dir = MODELS_PROJECT_ROOT / "conf"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), job_name="mattergen_finetune", version_base="1.1"):
        cfg = compose(config_name="finetune", overrides=overrides)
    return cfg


def run_finetune(options: FinetuneOptions) -> None:
    # Import heavy dependencies only when actually performing training.
    from omegaconf import OmegaConf, open_dict  # type: ignore
    from pytorch_lightning import LightningDataModule, Trainer  # type: ignore
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint  # type: ignore
    from pytorch_lightning.cli import SaveConfigCallback  # type: ignore
    from mattergen.diffusion.run import AddConfigCallback, SimpleParser, maybe_instantiate  # type: ignore
    from mattergen.scripts.finetune import init_adapter_lightningmodule_from_pretrained  # type: ignore
    output_dir_abs = options.output_dir.expanduser().resolve()
    output_dir_abs.mkdir(parents=True, exist_ok=True)
    os.environ["OUTPUT_DIR"] = str(output_dir_abs)
    for key, value in options.env.items():
        os.environ[key] = str(value)

    train_cache = options.data_train
    val_cache = options.data_val
    if not train_cache.exists() or not val_cache.exists():
        raise FileNotFoundError(f"Expected to find `train/` and `val/` under {options.data_train.parent}")
    if not options.model_dir.exists():
        raise FileNotFoundError(f"Non-exist pre-trained model path: {options.model_dir}.")

    overrides = _build_overrides(options)
    cfg = compose_config(overrides)

    torch.set_float32_matmul_precision("high")

    trainer: Trainer = maybe_instantiate(cfg.trainer, Trainer)
    datamodule: LightningDataModule = maybe_instantiate(cfg.data_module, LightningDataModule)

    if not trainer.logger:
        trainer.callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, LearningRateMonitor)]
        #trainer.callbacks.append(LearningRateMonitor(logging_interval="step", log_momentum=False))

    pl_module, lightning_module_cfg = init_adapter_lightningmodule_from_pretrained(cfg.adapter, cfg.lightning_module)

    with open_dict(cfg):
        cfg.lightning_module = lightning_module_cfg

    config_as_dict = OmegaConf.to_container(cfg, resolve=True)
    trainer.callbacks.append(SaveConfigCallback(parser=SimpleParser(), config=config_as_dict, overwrite=True))
    trainer.callbacks.append(AddConfigCallback(config_as_dict))

    # Ensure checkpoints are saved (best + last) similar to original mattergen trainer default.
    ckpt_dir = output_dir_abs / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    has_ckpt = any(isinstance(cb, ModelCheckpoint) for cb in trainer.callbacks)
    if not has_ckpt:
        trainer.callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="last",  # last epoch ckpt
                save_last=True,
                every_n_epochs=1,
                save_top_k=0,
            )
        )

    trainer.fit(model=pl_module, datamodule=datamodule, ckpt_path=str(options.resume) if options.resume else None)

    # Always emit a final checkpoint to a deterministic location.
    try:
        print("Saving final checkpoint...")
        final_ckpt = ckpt_dir / "last.ckpt"
        final_ckpt.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_ckpt), weights_only=True)
    except Exception as e:
        logging.warning(f"Failed to save final checkpoint: {e}")
    #return ckpt_dir


@Tools.register("mattergen_train")
def mattergen_train(
    *,
    model_path: Optional[Union[str, Path]],
    data_train: Union[str, Path],
    data_val: Optional[Union[str, Path]] = None,
    data_test: Optional[Union[str, Path]] = None,
    output_dir: Union[str, Path] = "./outputs",
    max_epochs: int = 10,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    num_workers: int = 4,
    devices: int = 1,
    accelerator: str = "auto",
    strategy: str = "auto",
    precision: str = "32",
    properties: List[str] = [],
    env: dict={},
    optional_args: list = [],
    resume: Optional[Union[str, Path]] = None,
    use_wandb: bool = False,
    skip: bool = False,
) -> tuple:
    """
    Run MatterGen finetuning using Python API (no subprocess).
    
    Args:
        model_path: Path to pretrained model directory
        data_train: Path to training data directory
        data_val: Path to validation data directory (optional)
        data_test: Path to test data directory (optional, currently unused)
        output_dir: Directory to save outputs and checkpoints
        max_epochs: Maximum number of training epochs
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        num_workers: Number of data loading workers
        devices: Number of devices to use
        accelerator: Accelerator type ('auto', 'gpu', 'cpu', etc.)
        strategy: Training strategy
        precision: Training precision ('32', '16', 'bf16', etc.)
        resume: Path to checkpoint to resume from (optional)
        use_wandb: Whether to use Weights & Biases logging
        skip: If True, skip training and use provided model
    
    Returns:
        (config_path, log_path, model_path, extra_outputs)
    """
    def _as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
        if p is None:
            return None
        return p if isinstance(p, Path) else Path(p)
    
    if skip:
        if model_path is None:
            raise ValueError("Model path must be provided when skip=True.")
        placeholder_script = Path("skip_config.yaml")
        placeholder_log = Path("skip_train.log")
        placeholder_script.write_text("skip: true\n")
        placeholder_log.write_text("Training skipped; using provided model.\n")
        return str(placeholder_script), str(placeholder_log), str(model_path), []
    
    data_train = _as_path(data_train)
    data_val = _as_path(data_val)
    data_test = _as_path(data_test)
    model_path = _as_path(model_path)
    output_dir = _as_path(output_dir)
    resume = _as_path(resume)
    
    if not data_train or not data_train.exists():
        raise ValueError("Training data path is required and must exist.")
    
    if model_path is None or not model_path.exists():
        raise ValueError("Model path is required and must exist for finetuning.")
    
    
    # Prepare FinetuneOptions
    options = FinetuneOptions(
        data_train=data_train,
        data_val=data_val,
        data_test=data_test,
        model_dir=model_path,
        output_dir=output_dir,
        max_epochs=max_epochs,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        precision=precision,
        resume=resume,
        properties=properties,
        env=env,
        use_wandb=use_wandb,
        optional_args=optional_args
    )
    
    # Run finetuning directly via Python API
    
    run_finetune(options)
    
    # Locate output files
    config_file = Path("config.yaml")
    log_file = Path("lightning_logs/version_0/metrics.csv")
    
    # Find the last checkpoint
    if not log_file.exists():
        log_file.write_text("Log file not found after training.\n")
        
    if config_file.exists():
        dest_config = Path(output_dir) / config_file.name
        dest_config.write_text(config_file.read_text())
    return str(config_file), str(log_file), str(output_dir), []
    

