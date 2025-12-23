"""Tool functions for MatterGen training and generation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from shutil import copy
from typing import Dict, List, Optional, Union
import logging
from ase.io import read, write
from ase import Atoms
import numpy as np
import json

from .base_tools import Tools

def dict_to_fire_args(config:Dict) -> list:
    """
    Convert a nested dictionary into a list of CLI arguments in the form of key=value.
    """
    return [f"--{key}={value}" for key, value in config.items()]

def _dict_to_hydra_args(config: Dict) -> list:
    """Convert nested dict to flat hydra-style key=value list."""

    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    return [f"{key}={value}" for key, value in flatten_dict(config)]


@Tools.register("mattergen_data")
def mattergen_data(
    ase_extxyz_file: Union[str,Path],
    mattergen_data: Union[str,Path],
    properties: Optional[List[str]] = [],
    )->str:
    """Transform ase.extxyz file to mattergen data format.
    
    Saves structures as separate .npy files:
    - cell.npy: (N, 3, 3) lattice vectors
    - pos.npy: (total_atoms, 3) concatenated positions
    - num_atoms.npy: (N,) atom counts per structure
    - atomic_numbers.npy: (total_atoms,) concatenated atomic numbers
    - structure_id.npy: (total_atoms,) structure index for each atom
    """
    
    atoms_ls = read(ase_extxyz_file,":")
    if isinstance(mattergen_data,str):
        mattergen_data = Path(mattergen_data)
    mattergen_data.mkdir(parents=True,exist_ok=True)
    num_structures = len(atoms_ls)
    
    # Prepare arrays
    cells = []
    positions = []
    num_atoms = []
    atomic_numbers = []
    structure_ids = []
    properties_data = {prop: {'values':[],'property_source_doc_id':prop,'origins':None} for prop in properties}
    for idx, atoms in enumerate(atoms_ls):
        cells.append(atoms.get_cell().array)
        positions.append(atoms.get_positions())
        num_atoms.append(len(atoms))
        atomic_numbers.append(atoms.get_atomic_numbers())
        if atoms.info.get("structure_id") is not None:
            structure_ids.append(atoms.info["structure_id"])
        else:
            structure_ids.append("%06d"%idx)
        for prop in properties:
            if prop in atoms.info:
                properties_data[prop]['values'].append(atoms.info[prop])
            else:
                properties_data[prop]['values'].append(None)
                logging.warning(f"Property '{prop}' not found in structure {idx}.")
    
    # Convert to numpy arrays
    cells_array = np.array(cells)  # (N, 3, 3)
    pos_array = np.vstack(positions)  # (total_atoms, 3)
    num_atoms_array = np.array(num_atoms, dtype=np.int32)  # (N,)
    atomic_numbers_array = np.concatenate(atomic_numbers)  # (total_atoms,)
    structure_id_array = np.array(structure_ids)#np.concatenate(structure_ids)  # (total_atoms,)
    
    # Save to disk
    np.save(mattergen_data / "cell.npy", cells_array)
    np.save(mattergen_data / "pos.npy", pos_array)
    np.save(mattergen_data / "num_atoms.npy", num_atoms_array)
    np.save(mattergen_data / "atomic_numbers.npy", atomic_numbers_array)
    np.save(mattergen_data / "structure_id.npy", structure_id_array)
    
    for prop in properties:
        with open(mattergen_data / f"{prop}.json", 'w') as f:
            json.dump(properties_data[prop], f, indent=4)
    
    logging.info(f"Saved {num_structures} structures to {mattergen_data}")
    logging.info(f"Total atoms: {len(pos_array)}")
    
    return str(mattergen_data)

@Tools.register("mattergen2ase")
def mattergen2ase(
    mattergen_data: Union[str, Path],
    ase_extxyz_file: Union[str, Path],
    properties: Optional[List[str]] = None,
) -> str:
    """Transform mattergen data format to ase.extxyz file.
    
    Reads structures from separate .npy files:
    - cell.npy: (N, 3, 3) lattice vectors
    - pos.npy: (total_atoms, 3) concatenated positions
    - num_atoms.npy: (N,) atom counts per structure
    - atomic_numbers.npy: (total_atoms,) concatenated atomic numbers
    - structure_id.npy: (N,) structure IDs
    - {property}.json: optional property data
    
    Args:
        mattergen_data: Path to directory containing MatterGen .npy files
        ase_extxyz_file: Output path for extxyz file
        properties: Optional list of property names to read from .json files
    
    Returns:
        Path to the created extxyz file
    """
    if isinstance(mattergen_data, str):
        mattergen_data = Path(mattergen_data)
    
    if not mattergen_data.exists():
        raise ValueError(f"MatterGen data directory does not exist: {mattergen_data}")
    
    # Load numpy arrays
    cells = np.load(mattergen_data / "cell.npy")  # (N, 3, 3)
    positions = np.load(mattergen_data / "pos.npy")  # (total_atoms, 3)
    num_atoms = np.load(mattergen_data / "num_atoms.npy")  # (N,)
    atomic_numbers = np.load(mattergen_data / "atomic_numbers.npy")  # (total_atoms,)
    
    # Load structure IDs if available
    structure_id_file = mattergen_data / "structure_id.npy"
    if structure_id_file.exists():
        structure_ids = np.load(structure_id_file)
    else:
        structure_ids = [f"{i:06d}" for i in range(len(num_atoms))]
    
    # Load properties if specified
    properties_data = {}
    if properties:
        for prop in properties:
            prop_file = mattergen_data / f"{prop}.json"
            if prop_file.exists():
                with open(prop_file, 'r') as f:
                    properties_data[prop] = json.load(f)
            else:
                logging.warning(f"Property file not found: {prop_file}")
    
    # Reconstruct individual Atoms objects
    atoms_list = []
    atom_offset = 0
    
    for i in range(len(num_atoms)):
        n_atoms = num_atoms[i]
        
        # Extract data for this structure
        cell = cells[i]
        pos = positions[atom_offset:atom_offset + n_atoms]
        nums = atomic_numbers[atom_offset:atom_offset + n_atoms]
        
        # Create Atoms object
        atoms = Atoms(
            numbers=nums,
            positions=pos,
            cell=cell,
            pbc=True
        )
        
        # Add structure ID to info
        atoms.info["structure_id"] = str(structure_ids[i])
        
        # Add properties to info
        for prop, prop_data in properties_data.items():
            if 'values' in prop_data and i < len(prop_data['values']):
                value = prop_data['values'][i]
                if value is not None:
                    atoms.info[prop] = value
        
        atoms_list.append(atoms)
        atom_offset += n_atoms
    
    # Write to extxyz file
    write(ase_extxyz_file, atoms_list)
    
    logging.info(f"Converted {len(atoms_list)} structures from {mattergen_data} to {ase_extxyz_file}")
    logging.info(f"Total atoms: {len(positions)}")
    
    return str(ase_extxyz_file)

@Tools.register("mattergen_train")
def mattergen_train(
    *,
    model_path: Optional[Union[str, Path]],
    data_train: Union[str, Path],
    data_val: Optional[Union[str, Path]] = None,
    data_test: Optional[Union[str, Path]] = None,
    arguments: Dict = {},
    additional_args: List = []                                                                                          ,
    custom_cmd: Optional[str] = None,
    is_finetune: bool = True,
    env: Dict = {},
    venv: Optional[str] = None,
    ) -> tuple:
    """
        Run MatterGen training or finetuning.

        Returns (train_script_name, log, model, extra_output).
    """

    def _as_path(p: Optional[Union[str, Path]]) -> Optional[Path]:
        if p is None:
            return None
        return p if isinstance(p, Path) else Path(p)

    skip = arguments.pop("skip", False)
    if skip:
        if model_path is None:
            raise ValueError("Model path must be provided when skip=True.")
        placeholder_script = Path("skip_config.yaml")
        placeholder_log = Path("skip_train.log")
        placeholder_script.write_text("skip: true\n")
        placeholder_log.write_text("Training skipped; using provided model.\n")
        return str(placeholder_script), str(placeholder_log), str(model_path), []

    data_train = _as_path(data_train)
    data_val = _as_path(data_val)
    data_test = _as_path(data_test)
    model_path = _as_path(model_path)

    if not data_train or not data_train.exists():
        raise ValueError("Training data path is required and must exist.")

    

    args_data = [f"data_module.train_dataset.cache_path={data_train}"]

    if data_val:
        args_data.append(f"data_module.val_dataset.cache_path={data_val}")
    else:
        logging.warning("Validation data not provided; skipping validation dataset.")
        args_data.append("~data_module.val_dataset")

    if data_test:
        args_data.append(f"data_module.test_dataset.cache_path={data_test}")
    else:
        logging.warning("Test data not provided; skipping test dataset.")
        args_data.append("~data_module.test_dataset")

    if custom_cmd:
        subprocess.run(custom_cmd.split(), check=True)
    else:
        if is_finetune:
            if model_path is None:
                raise ValueError("Model path is required for fine-tuning.")
            cmd = [
                "mattergen-finetune",
                f"adapter.model_path={str(model_path)}",
                *args_data,
                *_dict_to_hydra_args(arguments),
                *additional_args,
            ]
        else:
            cmd = [
                "mattergen-train",
                *_dict_to_hydra_args(arguments),
                *additional_args,
            ]
        if venv:
            cmd.insert(0, f"source {venv}/bin/activate && ")
            cmd_str = " ".join(cmd)
            subprocess.run(cmd_str, check=True, shell=True, env=env, executable="/bin/bash")
        else:
            subprocess.run(cmd, check=True, shell=False, env=env)

    outputs_path = Path("outputs")
    newest_dir = max(
        [p for p in outputs_path.rglob("*-*-*") if p.is_dir() and re.fullmatch(r"\d{2}-\d{2}-\d{2}", p.name)],
        key=lambda d: d.stat().st_mtime,
        default=None,
    )
    train_script_name = Path("config.yaml")
    log = Path("train.log")
    model_file = Path("last.ckpt")

    if newest_dir and newest_dir.is_dir():
        train_script_tmp = newest_dir / "config.yaml"
        if train_script_tmp.exists():
            copy(train_script_tmp, train_script_name)
        else:
            train_script_name.write_text("No available training script!")

        if (log_tmp := next(newest_dir.rglob("metrics.csv"), None)):
            copy(log_tmp, log)
        else:
            log.write_text("Not available log file!")

        if (ckpt := next(newest_dir.rglob("last.ckpt"), None)):
            copy(ckpt, model_file)
            return str(train_script_name), str(log), str(newest_dir), []
        else:
            raise RuntimeError("No checkpoint found in the latest directory.")

    raise RuntimeError("No output directory found after training.")


@Tools.register("mattergen_generate")
def mattergen_generate(
    *,
    model_path: Union[str, Path],
    results_dir: Union[str, Path] = "./",
    arguments: Dict = {},
    additional_args: List = [],
    custom_cmd: Optional[str] = None,
    env: Dict = {},
    venv: Optional[str] = None,
) -> tuple:
    """Run MatterGen generation using an existing model.

    Returns (generated_structures_path, extra_outputs).
    """

    arguments = arguments or {}
    additional_args = additional_args or []
    env = env or {}

    model_path = model_path if isinstance(model_path, Path) else Path(model_path)
    if not model_path.exists():
        raise ValueError("Model path must exist for generation.")

    results_dir = results_dir if isinstance(results_dir, Path) else Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if custom_cmd:
        subprocess.run(custom_cmd, check=True)
    else:
        cmd = [
            "mattergen-generate",
            f"{str(results_dir)}",
            f"--model_path={str(model_path)}",
            *dict_to_fire_args(arguments),
            *additional_args,
        ]
        logging.info("========Running generation=======")
        if venv:
            cmd.insert(0, f"source {venv}/bin/activate && ")
            cmd_str = " ".join(cmd)
            subprocess.run(cmd_str, check=True, shell=True, env=env, executable="/bin/bash")
        else:
            subprocess.run(cmd, check=True, shell=False, env=env)

    generated_crystals = results_dir / "generated_crystals.extxyz"
    if not generated_crystals.exists():
        raise RuntimeError("No generated crystals found.")

    extra_outputs: List[str] = []
    generated_crystals_traj = results_dir / "generated_trajectories.zip"
    if generated_crystals_traj.exists():
        extra_outputs.append(str(generated_crystals_traj))

    return str(generated_crystals), extra_outputs, {"message": "Generation completed successfully."}
