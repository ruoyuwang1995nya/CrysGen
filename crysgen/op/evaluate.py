from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    BigParameter,
    Artifact,
    Parameter
    )
from typing import Dict
from pathlib import Path


class SUNEvaluate(OP):
    """OP which evaluate and select new structures (It now replace previously seperated OPs)
    
    Args:
        structures (Artifact(Path)): structures file in xyz format.
        reference_dataset (Artifact(Path)): cached reference dataset. 
    """
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "structures": Artifact(Path),
                "reference_dataset": Artifact(Path,optional=True),
                "energies": Artifact(Path,optional=True),
                "properties": BigParameter(Dict[str,Path],optional=True,default={}),
                "config": Parameter(dict)
            },
        )
        
    @classmethod
    def get_output_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "selected_structures": Artifact(Path),  
                "selected_structures_properties": Artifact(Path),
                "results": BigParameter(dict),
                "energy_above_hull": Artifact(Path, optional=True),
                #"reference_dataset": Artifact(Path),
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip: OPIO,
        ) -> OPIO:
        from crysgen.tools.sun_eval.evaluate import sun_evaluate
        
        result = sun_evaluate(
            task_name=ip["task_name"],
            structures=ip["structures"],
            config=ip["config"],
            energies=ip.get("energies"),
            reference_dataset=ip.get("reference_dataset"),
            properties=ip.get("properties", {})
        )
        
        return OPIO(result)

