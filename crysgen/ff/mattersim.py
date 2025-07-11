from abc import ABC, abstractmethod
from typing import Optional,Union, List
from pathlib import Path
import numpy as np
from pymatgen.core import Structure
from crysgen.ff.potential import BasePotential
from crysgen.evaluation.utils.relaxation import relax_structures

@BasePotential.register("mattersim")
@BasePotential.register("MatterSim")
class MatterSim(BasePotential):
    def __init__(self):
        self.name = "MatterSim"
        super().__init__()
        
    def relax(
        self,
        structures: List[Structure], 
        output_path: str = "relaxed_structures.extxyz", 
        potential: Optional[str] = None,
        **kwargs
        ):
        """
        Relax the structure using the potential.
        Args:
            structures (List[Structure]): List of pymatgen Structure objects.
            potential (Optional[str]): Path to the potential file.
            output_path (str): Path to save the relaxed structure.
            **kwargs: Additional arguments for the relaxation process.
        """
        _, energies = relax_structures(
            structures=structures,
            potential_load_path=potential,
            output_path=output_path,
            **kwargs
        )
        np.save("energies.npy", energies)
        return output_path, "energies.npy",[] 
    
    def inference():
        pass
        
    