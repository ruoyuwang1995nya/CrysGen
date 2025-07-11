from typing import Optional, Union, Dict, List
from pathlib import Path
import subprocess
import warnings
from shutil import copy
import re
from crysgen.generator.model import BaseModel
from crysgen.generator.utils import dict_to_hydra_args
from crysgen.evaluation.reference.reference_dataset import ReferenceDataset

@BaseModel.register("mattergen")
@BaseModel.register("MatterGen")
class MatterGen(BaseModel):
    """
    The MatterGen model class for training and evaluation.
    """

    def __init__(self, model: Union[Path, str] = None):
        super().__init__(model)
        
    def load_model(self, model: Optional[Union[Path, str]] = None):
        self._model = model
    
    def get_data(self,
                 data_train: Union[Path,str] = None,
                 data_val: Optional[Union[Path,str]] = None,
                 data_test: Optional[Union[Path,str]] = None,
                 **kwargs
                 ):
        """
        Add dataset to the MatterGen model.
        """
        if data_train:
            if isinstance(data_train, str):
                data_train = Path(data_train)
            if not data_train.exists():
                raise ValueError(f"Training data path {data_train} does not exist.")
            self._data_train=str(data_train)
            print("check")
        if data_val:
            if isinstance(data_val, str):
                data_val = Path(data_val)
            if not data_val.exists():
                raise ValueError(f"Validation data path {data_val} does not exist.")
            self._data_val=str(data_val)
        if data_test:
            if isinstance(data_test, str):
                data_test = Path(data_test)
            if not data_test.exists():
                raise ValueError(f"Test data path {data_test} does not exist.")
            self._data_test=str(data_test)
    
    def train(self, 
              arguments: Optional[Dict] = {},
              additional_args: Optional[List] = [],
              custom_cmd: Optional[str] = None,
              is_finetune: Optional[bool] = True,
              env: Optional[Dict] = {},
              venv: Optional[str] = None,
              **kwargs # eat all other kwargs
              ):
        """
        Train the MatterGen model.
        """
        args_data=[]
        if not self.data_train:
            raise ValueError("Training data path is required.")
        args_data.append(f"data_module.train_dataset.cache_path={self.data_train}")
        
        if self.data_val:
            args_data.append(f"data_module.val_dataset.cache_path={self.data_val}")
        else:
            warnings.warn(
                "Validation data path is not provided. Proceeding without validation data. "
                "This may affect the evaluation of the model during training. Ensure that "
                "you provide validation data if you want to monitor validation metrics.",
                UserWarning
            )
            args_data.append(f"~data_module.val_dataset")
        if self.data_test:
            args_data.append(f"data_module.test_dataset.cache_path={self.data_test}")
        else:
            warnings.warn(
                "Test data path is not provided. Proceeding without test data. "
                "This may affect the evaluation of the model after training. Ensure that "
                "you provide test data if you want to monitor test metrics.",
                UserWarning
            )
            args_data.append(f"~data_module.test_dataset")
            
        if custom_cmd:
            subprocess.run(custom_cmd.split(), check=True)    
            
        else:
            if is_finetune:
                if self.model is None:
                    raise ValueError("Model path is required for fine-tuning.")
                print("Fine-tuning the MatterGen model...")
                try:
                    arguments=self.dict_to_hydra_args(arguments)
                    additional_args=additional_args
                    cmd= ['mattergen-finetune', f'adapter.model_path={str(self.model)}'] + args_data+ arguments + additional_args
                    if venv:
                        cmd.insert(0, f"source {venv}/bin/activate && ")
                    cmd_str = ' '.join(cmd)
                    subprocess.run(cmd_str, check=True,shell=True,env=env, executable='/bin/bash')
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Error during fine-tuning: {e}")
                print("Fine-tuning completed.")
            else:
                print("Training the MatterGen model...")
                try:
                    arguments=self.dict_to_hydra_args(arguments)
                    additional_args=additional_args
                    cmd=['mattergen-train'] + arguments + additional_args
                    if venv:
                        cmd.insert(0, f"source {venv}/bin/activate && ")
                    cmd_str = ' '.join(cmd)
                    subprocess.run(cmd_str, check=True,shell=True,env=env, executable='/bin/bash')
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"Error during training: {e}")
                print("Training completed.")
                
        outputs_path = Path("outputs")  # Assuming checkpoints are stored in a "checkpoints" directory
        newest_dir = max([p for p in outputs_path.rglob("*-*-*") if p.is_dir() and re.fullmatch(r"\d{2}-\d{2}-\d{2}", p.name)], 
                         key=lambda d: d.stat().st_mtime, default=None)
        train_script_name = "config.yaml"
        log = "train.log"
        model = "last.ckpt"
        if newest_dir and newest_dir.is_dir():
            print(newest_dir)
            train_script_tmp = newest_dir / "config.yaml"
            if train_script_tmp.exists():
                copy(train_script_tmp, Path (train_script_name))
            else:
                Path(train_script_name).write_text("No available training script!")
                
            if log_tmp:= next(newest_dir.rglob("metrics.csv"),None):
                copy(log_tmp, log)
            else:
                Path(log).write_text("Not available log file!")
            
            if  ckpt:= next(newest_dir.rglob("last.ckpt"),None):
                copy(ckpt,Path("last.ckpt"))
                return train_script_name, log, model, []
            else:
                raise RuntimeError("No checkpoint found in the latest directory.")   
        
    def generate(self, 
                arguments: Optional[Dict] = {},
                additional_args: Optional[List] = [],
                custom_cmd: Optional[str] = None,
                env: Optional[Dict] = {},
                venv: Optional[str] = None,
                results_dir: Optional[Union[Path, str]] = "./",
                **kwargs # eat all other kwargs
                 ):
        """
        Generate new samples using the MatterGen model.
        """
        if not self.model:
            raise ValueError("Model path is required for generation.")

        if isinstance(results_dir,str):
            results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        try:
            if custom_cmd:
                subprocess.run(custom_cmd, check=True)
            else:
                arguments=self.dict_to_fire_args(arguments)
                additional_args=additional_args
                print(str(self.model))
                cmd=['mattergen-generate', f'{str(results_dir)}',f'--model_path={str(self.model)}'] + arguments + additional_args
                if venv:
                    cmd.insert(0, f"source {venv}/bin/activate && ")
                cmd_str = ' '.join(cmd)
                subprocess.run(cmd_str, check=True,shell=True,env=env, executable='/bin/bash')
        except subprocess.CalledProcessError as e:
            print(f"Error during generation: {e}")

        generated_crystals = results_dir / "generated_crystals.extxyz"
        if generated_crystals.exists():
            self._stru_gen = str(generated_crystals)
            print(f"Generated crystals saved to: {self.stru_gen}")
        else:
            print("No generated crystals found.")
        
        extra_outputs = []            
        generated_crystals_traj = results_dir / "generated_trajectories.zip"
        if generated_crystals_traj.exists():
            self._stru_gen_traj = str(generated_crystals_traj)
            extra_outputs.append(self._stru_gen_traj)
        return self.stru_gen, extra_outputs
        
        
    @staticmethod
    def dict_to_hydra_args(config: Dict) -> list:
        """
        Convert a nested dictionary into a list of CLI arguments in the form of key=value.
        """
        def flatten_dict(d, parent_key=""):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key))
                else:
                    items.append((new_key, v))
            return items
        # Flatten the dictionary and format as key=value
        return [f"{key}={value}" for key, value in flatten_dict(config)]
    
    @staticmethod
    def dict_to_fire_args(config:Dict) -> list:
        """
        Convert a nested dictionary into a list of CLI arguments in the form of key=value.
        """
        return [f"--{key}={value}" for key, value in config.items()]