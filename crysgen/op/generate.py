from dflow.python import (
 OP,
 OPIO,
OPIOSign,   
BigParameter,
Artifact,
Parameter
)
from pathlib import Path
from typing import List, Dict, Union, Optional
from crysgen.utils import set_directory,make_path
from crysgen.tools import Tools
class Generate(OP):
    """Generate a structure using the trained model."""
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "model": Artifact(Path),
                "config": Parameter(dict),
            },
        )
        
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {   
                "generated_structures": Artifact(Path),
                "extra_outputs": Artifact(List[Path]),
                "results": BigParameter(Dict),
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip: OPIO
    ) -> OPIO:
        task_name = ip["task_name"]
        model_file=ip["model"]
        config=ip["config"]
        work_dir = Path(make_path(task_name))
        model_type = config.pop("model_type","foo")
        config.pop("num_tasks",None)
        with set_directory(work_dir):
            generate=Tools.get(model_type+"_generate")
            generated_structures, additional_outputs, results = generate(model_path=model_file, **config)
        return OPIO({
            "generated_structures": work_dir / generated_structures,
            "extra_outputs": [work_dir/ x for x in additional_outputs],
            "results": results
            
        })