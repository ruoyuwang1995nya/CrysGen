from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    BigParameter,
    Artifact,
    Parameter
    )
from pathlib import Path
from typing import List, Dict, Union, Optional, Any
from crysgen.utils import set_directory,make_path
from crysgen.tools import Tools
#from crysgen.generator.model import BaseModel

class Train(OP):
    """Train a model using the prepared dataset and ."""
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "config": Parameter(Dict[str,Any]),
                "base_model": Artifact(Path,optional=True),
                "training_data": Artifact(Path),
                "valid_data": Artifact(Path,optional=True),
                "test_data": Artifact(Path,optional=True)
            },
        )
        
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {
                "model": Artifact(Path),
                "report": Artifact(List[Path]),
            },
        )
    @OP.exec_sign_check 
    def execute(
        self,
        ip: OPIO,
    ) -> OPIO:
        task_name = ip["task_name"]
        config=ip["config"]
        base_model=ip["base_model"]
        training_data = ip.get("training_data")
        valid_data = ip.get("valid_data")
        test_data = ip.get("test_data")
        work_dir = Path(make_path(task_name))
        model_type=config.pop("model_type","foo")
        with set_directory(work_dir):
            train=Tools.get(model_type+"_train")
            train_script_name, log, model_file, extra_output = train(
                model_path=base_model,
                data_train=training_data,
                data_val=valid_data,
                data_test=test_data,
                **config
                )
        
        return OPIO(
            {
                "model": work_dir / model_file,
                "report": [work_dir / train_script_name,
                           work_dir / log] + [work_dir / x for x in extra_output]
            }
        )
        
    


