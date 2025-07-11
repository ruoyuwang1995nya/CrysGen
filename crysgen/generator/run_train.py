from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    BigParameter,
    Artifact,
    Parameter
    )
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from crysgen.utils import set_directory
from crysgen.generator.model import BaseModel

class RunTrain(OP):
    """Train a model using the prepared dataset and ."""
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "task_name": BigParameter(str),
                "task_path": Artifact(Path),
                "config": BigParameter(dict),
                "base_model": Artifact(Path,optional=True),
                "training_data": Artifact(Path),
                "test_data": Artifact(Path,optional=True),
                "valid_data": Artifact(Path,optional=True),
            },
        )
        
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {
                "log": Artifact(Path),
                "model": Artifact(Path),
                "script": Artifact(Path),
                "extra_outputs": Artifact(List[Path]),
            },
        )
    @OP.exec_sign_check 
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_name = ip["task_name"]
        #task_path = ip["task_path"]
        config=ip["config"]
        base_model=ip["base_model"]
        training_data = ip["training_data"]
        valid_data = ip["valid_data"]
        test_data = ip["test_data"]
        work_dir = Path(task_name)
        model_type=config["model_type"]
        with set_directory(work_dir):
            model=BaseModel.get_model(model_type)
            model=model()
            if base_model:
                model.load_model(base_model)
            model.get_data(
                data_train=training_data,
                data_val=valid_data,
                data_test=test_data,
            )
            print(model._data_train)
            train_script_name, log, model_file, extra_output = model.train(**config)
        return OPIO(
            {
                "script": work_dir / train_script_name,
                "model": work_dir / model_file,
                "log": work_dir / log,
                "extra_outputs": [work_dir / x for x in extra_output],
            }
        )
        
    


