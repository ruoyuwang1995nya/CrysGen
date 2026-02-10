from pathlib import Path
import numpy as np
import ase
from ase.io import write
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from crysgen.utils import set_directory
from crysgen.evaluation.metrics.evaluator import MetricsEvaluator
from crysgen.evaluation.reference.reference_dataset import ReferenceDataset
from crysgen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from crysgen.evaluation.utils.metrics_structure_summary import (
    get_metrics_structure_summaries
)


def sun_evaluate(
    task_name: str,
    structures: Path,
    config: dict,
    energies: Path = None,
    reference_dataset: Path = None,
    properties: dict = None
) -> dict:
    """Evaluate and select new structures.
    
    Args:
        task_name: Name of the task
        structures: Path to structures file in xyz format
        config: Configuration dictionary
        energies: Path to energies file (optional)
        reference_dataset: Path to cached reference dataset (optional)
        properties: Dictionary of property names to paths (optional)
        
    Returns:
        Dictionary containing:
            - selected_structures: Path to selected structures file
            - selected_structures_properties: Dict of property paths
            - results: Metric results dictionary
            - energy_above_hull: Path to energy above hull file (optional)
    """
    if properties is None:
        properties = {}
    
    if not reference_dataset:
        reference_dataset = config.get("reference_dataset")
        print("Using reference dataset from config.")
    if not reference_dataset:
        raise ValueError("Reference dataset must be provided either as input artifact or in config.")
    
    work_dir = Path(task_name)
    
    # read structures (in xyz format) and properties (in npy format) into structure_summaries
    try:
        ase_atoms = ase.io.read(structures, ":")
        structures_pmg = [AseAtomsAdaptor.get_structure(x) for x in ase_atoms]
    except Exception as e:
        print(f"Failed to read structure files!: {e}")
        raise RuntimeError("Failed to read structure files!") from e
    
    properties_dict = {k: np.load(v) for k, v in properties.items()}
    
    if energies:
        energies_arr = np.load(energies)
        assert len(energies_arr) == len(structures_pmg), "The number of energies and structures do not match!"
    else:
        energies_arr = None
    
    structure_summaries = get_metrics_structure_summaries(structures_pmg, energies_arr, properties_dict)
    
    # prepare reference dataset in ReferenceDataset format
    reference_dataset_obj = LMDBGZSerializer().deserialize(reference_dataset)
    
    with set_directory(work_dir):
        # create metrics evaluator
        evaluator = MetricsEvaluator.from_structure_summaries(
            structure_summaries,
            reference_dataset_obj,
            property_constraints=config.get("property_constraints")
        )
        
        metrics = get_metrics(
            evaluator,
            metrics=config.get("metrics", []),
        )
        
        metric_results = evaluator.compute_metrics(
            metrics=metrics,
            calc_pre_aggregate=config.get("calc_pre_aggregate", True)
        )
        
        # Access energy_above_hull directly from energy_capability if available
        energy_above_hull_values = None
        if hasattr(evaluator, 'energy_capability'):
            energy_above_hull_values = evaluator.energy_capability.energy_above_hull
            print(f"Energy above hull values: {energy_above_hull_values}")
            np.save("energy_above_hull.npy", energy_above_hull_values)
        
        # select structures
        selection_metrics = config.get("selection_criteria", [])
        default_mask = np.array([True] * len(structure_summaries))
        pre_aggregate_masks = [default_mask]
        
        for metric, res in tqdm(metric_results.items()):
            if metric in selection_metrics:
                pre_aggregate_metric = metric_results[metric].get("pre_aggregation_values")
                if pre_aggregate_metric is not None:
                    print("filtering with metric:", metric)
                    pre_aggregate_masks.append(pre_aggregate_metric)
        
        mask = np.logical_and.reduce(pre_aggregate_masks)
        selected_structure_summaries = [s for s, m in zip(structure_summaries, mask) if m]
        
        # write the structure to xyz format
        try:
            selected_structures = [s.entry.structure for s in selected_structure_summaries]
            selected_atoms = [AseAtomsAdaptor.get_atoms(s) for s in selected_structures]
            
            # Add energy_above_hull to info if available
            if energy_above_hull_values is not None:
                selected_energy_above_hull = energy_above_hull_values[mask]
                for atoms, e_hull in zip(selected_atoms, selected_energy_above_hull):
                    atoms.info['energy_above_hull'] = float(e_hull)
            
            write("selected_structures.extxyz", selected_atoms)
        except Exception as e:
            print(f"Failed to write structure files!: {e}")
            raise RuntimeError("Failed to write structure files!") from e
        
        selected_properties = {}
        for k in properties_dict.keys():
            arr = np.array([s.properties[k] for s in selected_structure_summaries])
            np.save(f"{k}.npy", arr)
            selected_properties[k] = f"{k}.npy"
        
        # Prepare energy_above_hull output
        energy_above_hull_path = None
        if energy_above_hull_values is not None:
            energy_above_hull_path = work_dir / "energy_above_hull.npy"
    
    return {
        "selected_structures": work_dir / "selected_structures.extxyz",
        "selected_structures_properties": {k: work_dir / v for k, v in selected_properties.items()},
        "results": metric_results,
        "energy_above_hull": energy_above_hull_path,
    }


def get_metrics(
    evaluator: MetricsEvaluator,
    metrics: list[str] = None
):
    """Get the metric types from a list of metric names.

    Args:
        evaluator (MetricsEvaluator): The evaluator to use.
        metrics (list[str], optional): A list of name. Defaults to None.

    Returns:
        list[object]: List of metric objects or "all"
    """
    if metrics is None:
        metrics = []
    
    if len(metrics) == 0:
        return "all"
    
    all_metrics = {metric.name: metric for metric in evaluator.available_metrics}
    
    metrics_list = []
    for k, v in all_metrics.items():
        if k in metrics:
            metrics_list.append(v)
    return metrics_list
