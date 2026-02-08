"""Tool functions for MatterSim relaxation and property prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Union
import logging
import numpy as np

from ase.io import read, write
from .base_tools import Tools


@Tools.register("mattersim_batch_relax")
def mattersim_relax(
    *,
    structures: Union[str, Path],
    potential: Union[str, Path],
    output_structures: Union[str, Path] = "./relaxed_structures.extxyz",
    device: str = "cpu",
    filter: str = "ExpCellFilter",
    fmax: float = 0.05,
    optimizer: str = "FIRE",
    **kwargs
) -> tuple:
    """Relax crystal structures using MatterSim potential.
    
    Args:
        structures: Path to input structures file (extxyz format)
        potential: Path to MatterSim potential checkpoint
        output_structures: Path to save relaxed structures (extxyz format)
        device: Device to use ('cpu', 'cuda', etc.)
        filter: ASE filter type ('ExpCellFilter', 'FrechetCellFilter', etc.)
        fmax: Maximum force convergence criterion
        optimizer: ASE optimizer to use ('FIRE', 'BFGS', 'LBFGS', etc.)
        **kwargs: Additional arguments passed to BatchRelaxer
    
    Returns:
        (relaxed_structures_path, energies_file, extra_outputs)
    """
    from mattersim.applications.batch_relax import BatchRelaxer
    from mattersim.forcefield.potential import Potential
    
    # Convert paths
    structures = Path(structures) if isinstance(structures, str) else structures
    potential = Path(potential) if isinstance(potential, str) else potential
    output_structures = Path(output_structures) if isinstance(output_structures, str) else output_structures
    if not structures.exists():
        raise ValueError(f"Input structures file not found: {structures}")
    if not potential.exists():
        raise ValueError(f"Potential checkpoint not found: {potential}")
    
    # Read input structures
    logging.info(f"Reading structures from {structures}")
    atoms_list = read(structures, ":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    
    logging.info(f"Loaded {len(atoms_list)} structures")
    
    # Load potential
    logging.info(f"Loading potential from {potential}")
    potential = Potential.from_checkpoint(
        device=device,
        load_path=str(potential),
        load_training_state=False
    )
    
    # Set up relaxer
    relaxer_kwargs = {
        "filter": filter,
        "fmax": fmax,
        #"steps": steps,
        "optimizer": optimizer,
        **kwargs
    }
    
    logging.info(f"Initializing BatchRelaxer with settings: {relaxer_kwargs}")
    batch_relaxer = BatchRelaxer(potential=potential, **relaxer_kwargs)
    
    # Perform relaxation
    logging.info("Starting relaxation...")
    relaxation_trajectories = batch_relaxer.relax(atoms_list)
    
    # Extract relaxed structures and energies
    relaxed_atoms = [t[-1] for t in relaxation_trajectories.values()]
    total_energies = np.array([a.info.get("total_energy", np.nan) for a in relaxed_atoms])
    
    # Save relaxed structures
    output_structures.parent.mkdir(parents=True, exist_ok=True)
    write(output_structures, relaxed_atoms, format="extxyz")
    logging.info(f"Relaxed structures saved to {output_structures}")
    
    # Save energies
    energies_file = output_structures.parent / "energies.npy"
    np.save(energies_file, total_energies)
    logging.info(f"Total energies saved to {energies_file}")
    
    return str(output_structures), str(energies_file), {
        "num_structures": len(relaxed_atoms),
        "mean_energy": float(np.mean(total_energies)),
        "min_energy": float(np.min(total_energies)),
        "max_energy": float(np.max(total_energies)),
    }
