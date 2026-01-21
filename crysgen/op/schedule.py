from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    BigParameter,
    Parameter,
    Artifact,
    )

from typing import List, Dict
from pathlib import Path
        
class Schedule(OP):
    """Generate a structure using the trained model."""
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "iter_id": Parameter(int,default=0),
                "iter_num": Parameter(int,default=1),
                "iter_data": Artifact(List[Path],optional=True),
                "incoming_data": Artifact(List[Path],optional=True),
                "config": Parameter(dict),
            },
        )
        
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {   
                "iter_data": Artifact(List[Path],optional=True),
                "iter_id": Parameter(int),
                "stop": Parameter(bool,optional=False)
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip: OPIO
    ) -> OPIO:
        iter_num = ip["iter_id"]
        max_iter= ip["iter_num"]
        iter_data=ip.get("iter_data",[])
        incoming_data=ip.get("incoming_data",[])
        config=ip["config"]
        if incoming_data:
            iter_data.extend(incoming_data) 
        if iter_num >= 0: 
            iter_num+=1      
        
        stop= False
        if iter_num >= max_iter:
            stop= True
        
        return OPIO({
            "iter_data": iter_data,
            "iter_id": iter_num,
            "stop": stop
        })