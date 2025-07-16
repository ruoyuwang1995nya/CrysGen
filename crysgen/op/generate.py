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
from crysgen.utils import set_directory
from crysgen.generator.model import BaseModel
from crysgen.evaluation.utils.metrics_structure_summary import MetricsStructureSummary
class Generate(OP):
    """Generate a structure using the trained model."""
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "task_name": BigParameter(str),
                "model": Artifact(Path),
                "config": BigParameter(dict),
            },
        )
        
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {   
                "generated_structures": Artifact(Path),
                "extra_outputs": Artifact(List[Path])
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
        work_dir = Path(task_name)
        model_type = config.pop("model_type")
        with set_directory(work_dir):
            model=BaseModel.get_model(model_type)
            model=model()
            model.load_model(model_file)
            generated_structures, additional_outputs = model.generate(**config)
        return OPIO({
            "generated_structures": work_dir / generated_structures,
            "extra_outputs": [work_dir/ x for x in additional_outputs] 
            
        })