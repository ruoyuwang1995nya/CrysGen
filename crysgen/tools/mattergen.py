"""Tool functions for MatterGen training and generation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from shutil import copy
from typing import Dict, List, Optional, Union
import logging

from .base_tools import Tools

def dict_to_fire_args(config:Dict) -> list:
    """
    Convert a nested dictionary into a list of CLI arguments in the form of key=value.
    """
    return [f"--{key}={value}" for key, value in config.items()]

def _dict_to_hydra_args(config: Dict) -> list:
    """Convert nested dict to flat hydra-style key=value list."""

    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    return [f"{key}={value}" for key, value in flatten_dict(config)]


@Tools.register("mattergen_train")
def mattergen_train(
    *,
    model_path: Optional[Union[str, Path]],
    data_train: Union[str, Path],
    data_val: Optional[Union[str, Path]] = None,
    data_test: Optional[Union[str, Path]] = None,
    arguments: Dict = {},
    additional_args: List = []                                                                                          ,
    custom_cmd: Optional[str] = None,
    is_finetune: bool = True,
    env: Dict = {},
    venv: Optional[str] = None,
    ) -> tuple:
    """
        Run MatterGen training or finetuning.

        Returns (train_script_name, log, model, extra_output).
    """

    def _as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
        if p is None:
            return None
        return p if isinstance(p, Path) else Path(p)

    skip = arguments.pop("skip", False)
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

    if not data_train or not data_train.exists():
        raise ValueError("Training data path is required and must exist.")

    

    args_data = [f"data_module.train_dataset.cache_path={data_train}"]

    if data_val:
        args_data.append(f"data_module.val_dataset.cache_path={data_val}")
    else:
        logging.warning("Validation data not provided; skipping validation dataset.")
        args_data.append("~data_module.val_dataset")

    if data_test:
        args_data.append(f"data_module.test_dataset.cache_path={data_test}")
    else:
        logging.warning("Test data not provided; skipping test dataset.")
        args_data.append("~data_module.test_dataset")

    if custom_cmd:
        subprocess.run(custom_cmd.split(), check=True)
    else:
        if is_finetune:
            if model_path is None:
                raise ValueError("Model path is required for fine-tuning.")
            cmd = [
                "mattergen-finetune",
                f"adapter.model_path={str(model_path)}",
                *args_data,
                *_dict_to_hydra_args(arguments),
                *additional_args,
            ]
        else:
            cmd = [
                "mattergen-train",
                *_dict_to_hydra_args(arguments),
                *additional_args,
            ]
        if venv:
            cmd.insert(0, f"source {venv}/bin/activate && ")
            cmd_str = " ".join(cmd)
            subprocess.run(cmd_str, check=True, shell=True, env=env, executable="/bin/bash")
        else:
            subprocess.run(cmd, check=True, shell=False, env=env)

    outputs_path = Path("outputs")
    newest_dir = max(
        [p for p in outputs_path.rglob("*-*-*") if p.is_dir() and re.fullmatch(r"\d{2}-\d{2}-\d{2}", p.name)],
        key=lambda d: d.stat().st_mtime,
        default=None,
    )
    train_script_name = Path("config.yaml")
    log = Path("train.log")
    model_file = Path("last.ckpt")

    if newest_dir and newest_dir.is_dir():
        train_script_tmp = newest_dir / "config.yaml"
        if train_script_tmp.exists():
            copy(train_script_tmp, train_script_name)
        else:
            train_script_name.write_text("No available training script!")

        if (log_tmp := next(newest_dir.rglob("metrics.csv"), None)):
            copy(log_tmp, log)
        else:
            log.write_text("Not available log file!")

        if (ckpt := next(newest_dir.rglob("last.ckpt"), None)):
            copy(ckpt, model_file)
            return str(train_script_name), str(log), str(newest_dir), []
        else:
            raise RuntimeError("No checkpoint found in the latest directory.")

    raise RuntimeError("No output directory found after training.")


@Tools.register("mattergen_generate")
def mattergen_generate(
    *,
    model_path: Union[str, Path],
    results_dir: Union[str, Path] = "./",
    arguments: Dict = {},
    additional_args: List = [],
    custom_cmd: Optional[str] = None,
    env: Dict = {},
    venv: Optional[str] = None,
) -> tuple:
    """Run MatterGen generation using an existing model.

    Returns (generated_structures_path, extra_outputs).
    """

    arguments = arguments or {}
    additional_args = additional_args or []
    env = env or {}

    model_path = model_path if isinstance(model_path, Path) else Path(model_path)
    if not model_path.exists():
        raise ValueError("Model path must exist for generation.")

    results_dir = results_dir if isinstance(results_dir, Path) else Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if custom_cmd:
        subprocess.run(custom_cmd, check=True)
    else:
        cmd = [
            "mattergen-generate",
            f"{str(results_dir)}",
            f"--model_path={str(model_path)}",
            *dict_to_fire_args(arguments),
            *additional_args,
        ]
        logging.info("========Running generation=======")
        if venv:
            cmd.insert(0, f"source {venv}/bin/activate && ")
            cmd_str = " ".join(cmd)
            subprocess.run(cmd_str, check=True, shell=True, env=env, executable="/bin/bash")
        else:
            subprocess.run(cmd, check=True, shell=False, env=env)

    generated_crystals = results_dir / "generated_crystals.extxyz"
    if not generated_crystals.exists():
        raise RuntimeError("No generated crystals found.")

    extra_outputs: List[str] = []
    generated_crystals_traj = results_dir / "generated_trajectories.zip"
    if generated_crystals_traj.exists():
        extra_outputs.append(str(generated_crystals_traj))

    return str(generated_crystals), extra_outputs, {"message": "Generation completed successfully."}
