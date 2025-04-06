from crysgen.train.model import BaseModel
from crysgen.train.utils import dict_to_mattergen_args
from typing import Optional, Union, Dict, List
from pathlib import Path
import subprocess
import warnings

@BaseModel.register("mattergen")
@BaseModel.register("MatterGen")
class MatterGen(BaseModel):
    """
    The MatterGen model class for training and evaluation.
    """

    def __init__(self, model: Union[Path, str] = None):
        super().__init__(model)
        
    
    def get_data(self,
                 data_train: Optional[Union[Path, str]] = None,
                 data_val: Optional[Union[Path, str]] = None,
                 data_test: Optional[Union[Path, str]] = None,
                 **kwargs):
        """
        Convert a dictionary to MatterGen arguments.
        """
        if data_train:
            if isinstance(data_train, str):
                data_train = Path(data_train)
            if not data_train.exists():
                raise ValueError(f"Training data path {data_train} does not exist.")
            self._data_train=str(data_train)
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
              config: Optional[Dict] = {},
              additional_args: Optional[List] = [],
              custom_cmd: Optional[str] = None,
              is_finetune: Optional[bool] = True,
              env: Optional[Dict] = {},
              venv: Optional[str] = None,
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
            

        # Implement the training logic here
        if is_finetune:
            # Fine-tuning logic
            if self.model is None:
                raise ValueError("Model path is required for fine-tuning.")
            
            print("Fine-tuning the MatterGen model...")
            try:
                if custom_cmd:
                    subprocess.run(custom_cmd.split(), check=True)
                else:
                    # parse arguments and run the command
                    arguments=self.dict_to_mattergen_args(config)
                    additional_args=additional_args
                    # parse additional commands
                    cmd= ['mattergen-finetune', f'adapter.model_path={str(self.model)}'] + args_data+ arguments + additional_args
                    # Activate the virtual environment
                    if venv:
                        cmd.insert(0, f"source {venv}/bin/activate && ")
                    cmd_str = ' '.join(cmd)
                    subprocess.run(cmd_str, check=True,shell=True,env=env, executable='/bin/bash')
                
            except subprocess.CalledProcessError as e:
                print(f"Error during fine-tuning: {e}")

            # Perform fine-tuning steps
            # Example: Update model parameters using provided data
            print("Fine-tuning completed.")
            
        else:
            # Training logic
            print("Training the MatterGen model...")
            try:
                if custom_cmd:
                    subprocess.run(custom_cmd.split(), check=True)
                else:
                    # parse arguments and run the command
                    arguments=self.dict_to_mattergen_args(config)
                    additional_args=additional_args
                    # parse additional commands
                    cmd=['mattergen-train'] + arguments + additional_args
                    subprocess.run(cmd, check=True)
                
            except subprocess.CalledProcessError as e:
                print(f"Error during training: {e}")
            # Perform training steps
            # Example: Update model parameters using provided data
            print("Training completed.")
            
        # extract checkpoints
        # Find the newest directory with the format YY-MM-DD/hh-mm-ss
        outputs_path = Path("outputs")  # Assuming checkpoints are stored in a "checkpoints" directory
        newest_dir = max(outputs_path.glob("*/**"), key=lambda d: d.stat().st_mtime, default=None)

        if newest_dir and newest_dir.is_dir():
            # Check if the directory contains a last.ckpt in newest_dir
            for ckpt in newest_dir.rglob("last.ckpt"):
                print(f"Found checkpoint: {ckpt}")
                return ckpt
            return None
        else:
            print("No valid checkpoint directories found.")
        
    
    def generate(self, *args, **kwargs):
        """
        Generate new samples using the MatterGen model.
        """
        pass
    
    @staticmethod
    def dict_to_mattergen_args(config: Dict) -> list:
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
    