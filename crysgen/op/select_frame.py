from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    BigParameter,
    Artifact,
    Parameter
    )
from pathlib import Path
from typing import List,Dict, Union
from crysgen.tools import Tools
import logging


class SelectFrame(OP):
    """
    OP filtering and selecting structures based on elements and basic properties.
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
                "structures": Artifact(List[Path]),
                "config": Parameter(dict)
            },
        )
        
    @classmethod
    def get_output_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "selected_structures": Artifact(Path),
                "selected_indices": BigParameter(list),
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip:OPIO,
        ) -> OPIO:
        from ase.io import read, write
        structures = ip["structures"]
        config = ip["config"]
        print(config)
        selectors = config.get("selectors", {})

        # Accept a single multi-frame file or a list of files and aggregate all frames
        atoms_ls: List = []
        if isinstance(structures, (list, tuple)):
            for path in structures:
                atoms_ls.extend(read(path, ":"))
        else:
            atoms_ls = read(structures, ":")

        masks: List[List[bool]] = []
        for selector_name, selector_cfg in selectors.items():
            print(f"Applying selector: {selector_name} with config: {selector_cfg}")
            _, mask = Tools.get(selector_name)(atoms_ls, **selector_cfg)
            masks.append(mask)

        # Combine masks: keep entry only if all masks are True; if no masks, keep all
        if masks:
            combined_mask = [all(bits) for bits in zip(*masks)]
        else:
            combined_mask = [True] * len(atoms_ls)

        selected_indices = [idx for idx, flag in enumerate(combined_mask) if flag]
        selected_atoms = [atoms_ls[idx] for idx in selected_indices]

        output_path = Path("selected_structures.extxyz")
        if selected_atoms:
            print(f"Selected {len(selected_atoms)} structures out of {len(atoms_ls)}")
            write(output_path, selected_atoms, format="extxyz")
        else:
            # still write an empty file to satisfy downstream expectations
            output_path.write_text("")

        return OPIO(
            {
                "selected_structures": output_path,
                "selected_indices": selected_indices,
            }
        )


class SelectFrameVasp(OP):
    """
    OP filtering and selecting structures based on elements and basic properties.
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
                "structures": Artifact(List[Path]),
                "config": BigParameter(dict)
            },
        )
        
    @classmethod
    def get_output_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "selected_structures": Artifact(Path),
                "selected_indices": BigParameter(list),
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip:OPIO,
        ) -> OPIO:
        from pymatgen.io.vasp.outputs import Vasprun
        outcar_ls = ip["structures"]
        config = ip["config"]
        selectors = config.get("selectors", {})
        
        atoms_ls = []
        #energies_ls=[]
        #properties: Dict[str, List] = {}
        
        for outcar in outcar_ls:
            try:
                vasprun = Vasprun(str(outcar), parse_potcar_file=False, occu_tol=1e-8)
                if not vasprun.converged_electronic:
                    raise ValueError(f"VASP calculation not converged for {outcar}, skipping.")
                
                
                
                #for selector in selectors:
                    
                
                
            except Exception as e:
                print(f"Failed to read VASP output file {outcar}!: {e}")
                continue
                
            
        
        atoms_ls = read(structures, ":")

        masks: List[List[bool]] = []
        for selector_name, selector_cfg in selectors.items():
            _, mask = Tools.get(selector_name)(atoms_ls, **selector_cfg)
            masks.append(mask)

        # Combine masks: keep entry only if all masks are True; if no masks, keep all
        if masks:
            combined_mask = [all(bits) for bits in zip(*masks)]
        else:
            combined_mask = [True] * len(atoms_ls)

        selected_indices = [idx for idx, flag in enumerate(combined_mask) if flag]
        selected_atoms = [atoms_ls[idx] for idx in selected_indices]

        output_path = Path("selected_structures.vasp")
        if selected_atoms:
            write(output_path, selected_atoms, format="vasp")
        else:
            # still write an empty file to satisfy downstream expectations
            output_path.write_text("")

        return OPIO(
            {
                "selected_structures": output_path,
                "selected_indices": selected_indices,
            }
        )