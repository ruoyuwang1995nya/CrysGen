from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    Artifact,
    Parameter,
    BigParameter
    )
from pathlib import Path
from crysgen.utils import set_directory
from crysgen.tools import Tools
from typing import Dict, Any


class RelaxFF(OP):
    """Relax a structure using the trained model."""
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {   
                "task_name": Parameter(str,default="relaxation"),
                "structures": Artifact(Path),
                "model": Artifact(Path,optional=True),
                "config": Parameter(dict),
            },
        )
        
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {   
                "relaxed_structures": Artifact(Path),
                "energies": Artifact(Path),
                "extra_outputs": BigParameter(Dict[str,Any]),
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip: OPIO
    ) -> OPIO:
        # read a list of pymatgen.Structures
        structures = ip["structures"]
        #print(structures)
        config=ip["config"]#.get("ff_relax",{})
        model_file=ip["model"]
        task_name = ip["task_name"]
        work_dir = Path(task_name)
        relaxer=config.pop("relaxer")
        with set_directory(work_dir):
            try:
                relaxed_structure, energies, extra_outputs = Tools.get(relaxer)(
                    structures=structures,
                    potential=model_file,
                    **config
            )
            except Exception as e:
                print(f"Relaxation failed: {e}")
                raise RuntimeError("Relaxation failed!") from e
            
        return OPIO({
            "relaxed_structures": work_dir / relaxed_structure,
            "energies": work_dir / energies,
            "extra_outputs": extra_outputs,
        })