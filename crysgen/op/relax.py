from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    BigParameter,
    Artifact,
    Parameter
    )
from pathlib import Path
from typing import List
import ase
from pymatgen.io.ase import AseAtomsAdaptor
from crysgen.utils import set_directory
from crysgen.ff.potential import BasePotential

class RelaxFF(OP):
    """Relax a structure using the trained model."""
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {   
                "task_name": BigParameter(str),
                "structures": Artifact(Path),
                "model": Artifact(Path,optional=True),
                "config": BigParameter(dict),
            },
        )
        
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {   
                "relaxed_structure": Artifact(Path),
                "energies": Artifact(Path),
                "extra_outputs": Artifact(List[Path])
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip: OPIO
    ) -> OPIO:
        # read a list of pymatgen.Structures
        original_structures = ip["structures"]
        config=ip["config"]
        model_file=ip["model"]
        task_name = ip["task_name"]
        work_dir = Path(task_name)
        try:
            ase_atoms = ase.io.read(original_structures, ":")
            structures = [AseAtomsAdaptor.get_structure(x) for x in ase_atoms]
        except Exception as e:
            print(f"Failed to read structure files!: {e}")
            raise RuntimeError("Failed to read structure files!") from e
        
        potential_type=config.pop("potential_type")
        potential=BasePotential.get_model(potential_type)
        potential=potential()
        with set_directory(work_dir):
            try:
                relaxed_structure, energies, extra_outputs = potential.relax(
                    structures=structures,
                    potential=model_file,
                    **config
            )
            except Exception as e:
                print(f"Relaxation failed: {e}")
                raise RuntimeError("Relaxation failed!") from e
            
        return OPIO({
            "relaxed_structure": work_dir / relaxed_structure,
            "energies": work_dir / energies,
            "extra_outputs": [work_dir / x for x in extra_outputs]
        })