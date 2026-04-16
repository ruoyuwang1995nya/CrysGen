"""
Specialized OP for solid electrolyte evaluation.    
"""

from dflow.python import (
    OP, 
    OPIO, 
    OPIOSign, 
    BigParameter,
    Artifact,
    Parameter,
    TransientError
    )
from pathlib import Path
from typing import List, Dict, Any
from ase.io import write,read
from crysgen.utils import set_directory
from crysgen.tools.ase_calculator import CalculatorWrapper
from crysgen.tools.ase_md import MDRunner
import numpy as np
import logging
import json
from copy import copy

class SelectFrameVaspSolidElectrolyte(OP):
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
                #"config": BigParameter(dict)
            },
        )
        
    @classmethod
    def get_output_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "selected_structures": Artifact(Path),
                "selected_indices": BigParameter(list),
                "energies": Artifact(Path),
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip:OPIO,
        ) -> OPIO:
        from pymatgen.io.vasp.outputs import Vasprun
        from pymatgen.io.ase import AseAtomsAdaptor
        from ase.io import write
        outcar_ls = ip["structures"]
        #config = ip["config"]
        #selectors = config.get("selectors", {})
        
        atoms_ls = []
        selected_indices: List[int] = []
        energies: List[float] = []
        
        for idx, outcar in enumerate(outcar_ls):
            try:
                vr = Vasprun(str(outcar), parse_potcar_file=False, occu_tol=1e-8)
                if not vr.converged_electronic:
                    raise ValueError(f"VASP calculation not converged for {outcar}, skipping.")
                
                bg = vr.eigenvalue_band_properties
                gap, _, _, _ = bg

                final_structure = vr.final_structure
                atoms = AseAtomsAdaptor.get_atoms(final_structure)
                atoms.info["bandgap"] = gap
                atoms.info["energy"] = vr.final_energy
                
                if gap >= 0.5:  # insulator
                    atoms_ls.append(atoms)
                    selected_indices.append(idx)
                    energies.append(vr.final_energy)

            except Exception as e:
                print(f"Failed to read VASP output file {outcar}!: {e}")
                continue
                
        output_path = Path("selected_structures.extxyz")
        if atoms_ls:
            write(output_path, atoms_ls, format="extxyz")
            logging.info(f"Selected {len(atoms_ls)} structures out of {len(outcar_ls)} based on criteria.")
        else:
            output_path.write_text("")
            logging.info("No structures selected based on criteria.")

        energies_path = Path("energies.npy")
        np.save(energies_path, np.array(energies, dtype=float))
        return OPIO(
            {
                "selected_structures": output_path,
                "selected_indices": selected_indices,
                "energies": energies_path,
            }
        )

class IonMD(OP):
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "structure": Artifact(Path),
                "model": Artifact(Path),
                "config": Parameter(Dict),
                #"stages": BigParameter(List[Dict]),
            },
        )
        
    @classmethod
    def get_output_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "traj": Artifact(Path),
                "results": Artifact(Path),
                "additional_results":Artifact(List[Path])
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip: OPIO
    ) -> OPIO:
        structure_path = ip["structure"]
        #stages = ip["stages"]
        model = ip["model"]
        config: Dict[str, Any] = ip["config"]
        stages: List[Dict[str, Any]] = config.pop("stages", [])
        task_name = ip["task_name"]
        
        # get atoms
        atoms = read(structure_path)
        
        # Create supercell if specified in config
        supercell = config.get("supercell", None)
        print(supercell)
        if supercell:
            from ase.build import make_supercell
            if isinstance(supercell, list) and len(supercell) == 3:
                # Simple [nx, ny, nz] format
                atoms = atoms * supercell
                logging.info(f"Created supercell {supercell}, new cell has {len(atoms)} atoms")
                print(f"Created supercell {supercell}, new cell has {len(atoms)} atoms")
            elif isinstance(supercell, list) and len(supercell) == 9:
                # 3x3 transformation matrix (flattened)
                matrix = np.array(supercell).reshape(3, 3)
                atoms = make_supercell(atoms, matrix)
                logging.info(f"Created supercell with transformation matrix, new cell has {len(atoms)} atoms")
            else:
                logging.warning(f"Invalid supercell format: {supercell}, skipping supercell creation")
        
        with set_directory(Path(task_name)):
            calc_cfg = dict(config.get("calculator", {}))
            calc_style = calc_cfg.pop("style", "mattersim")
            calc = CalculatorWrapper.get_calculator(calc_style)
            calc = calc().create(model_path=str(model), **calc_cfg)

            runner = MDRunner.from_atoms(atoms)
            runner.calc = calc
            additional_results = []
            # start md simulation
            try:
                log_dir = config.get("log_dir")
                traj_dir = config.get("traj_dir")
                res=runner.run_md_ion_stages(
                    stages=stages,
                    log_dir=log_dir,
                    traj_dir=traj_dir,
                )
                plot_path = Path("msd_plot.png").resolve()
                if plot_path.exists():
                    additional_results.append(plot_path)
                results_path = Path("md_results.json").resolve()
                with open(results_path, "w") as f:
                    json.dump(res["analysis"], f, indent=4)
            except Exception as e:
                raise TransientError(f"MD simulation failed: {e}")

        return OPIO({
            "traj": res["last_traj"],
            "results": results_path,
            "additional_results": additional_results
        })
    
class SelectFrameIonMD(OP):
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "structures": Artifact(List[Path]),
                "results": Artifact(List[Path]),
                "config": Parameter(Dict)  # config for criteria
            },
        )
        
    @classmethod
    def get_output_sign(cls)-> OPIOSign:
        return OPIOSign(
            {
                "selected_structures": Artifact(List[Path]),
                "selected_results": Artifact(Path)
            },
        )
        
    @OP.exec_sign_check
    def execute(
        self, 
        ip: OPIO
    ) -> OPIO:
        structures: List[Path] = ip["structures"]
        config: Dict[str, Any] = ip["config"]
        results: List[Path] = ip["results"]
        selected_structures = []
        selected_results = {}
        li_threshold = config.get("li_above", 1e-6)  # Li diffusion threshold in cm^2/s
        other_floor = config.get("other_floor", 1e-8)
        other_divisor = config.get("other_divisor", 1000)  # threshold_other = max(other_floor, li_diff/other_divisor)
        
        for idx, (structure, res_path) in enumerate(zip(structures, results)):
            try:
                with open(res_path, "r") as f:
                    res = json.load(f)
                diff_data = res.get("diff", {})
                li_diff_entry = diff_data.get("Li")
                if not li_diff_entry:
                    continue

                li_diff = li_diff_entry[0]
                other_diffs = [diff_data[ele][0] for ele in diff_data if ele != "Li" and diff_data.get(ele)]

                if not other_diffs:
                    continue

                max_other_diff = max(other_diffs)
                threshold_other = max(other_floor, li_diff / other_divisor)

                if li_diff > li_threshold and max_other_diff < threshold_other:
                    atoms = read(structure)
                    composition = atoms.get_chemical_formula(mode='hill', empirical=True)
                    ratio = li_diff / max_other_diff if max_other_diff > 0 else float("inf")

                    res_tmp = copy(res)
                    res_tmp.update({
                        "formula": composition,
                        "li_diff": li_diff,
                        "max_other_diff": max_other_diff,
                        "ratio": ratio,
                        "threshold_other": threshold_other,
                        "idx": f"{idx:06d}",
                    })

                    name = f"selected_{idx:06d}_{composition}.extxyz"
                    atoms.info.update({
                        "solid_electrolyte": "yes",
                    })
                    write(name, atoms)
                    selected_results[f"{idx:06d}"] = res_tmp
                    selected_structures.append(Path(name))
                    logging.info(
                        f"Task {idx} {composition}: Li={li_diff:.2e} cm²/s, "
                        f"max_other={max_other_diff:.2e} cm²/s, ratio={ratio:.1f}, "
                        f"threshold={threshold_other:.2e}"
                    )
            except Exception as e:
                logging.warning(f"Failed to process structure {structure}!: {e}")
                continue
        logging.info(f"Selected {len(selected_structures)} structures based on ion diffusion criteria.")
        with open("selected_results.json", "w") as f:
            json.dump(selected_results, f, indent=4)
        return OPIO({
            "selected_structures": selected_structures,
            "selected_results": Path("selected_results.json")
        })