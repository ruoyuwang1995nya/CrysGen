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
        ip:OPIO,
        ) -> OPIO:
        structures = ip["structures"]
        config = ip["config"]
        energies = ip["energies"]
        reference_dataset = ip.get("reference_dataset")
        if not reference_dataset:
            reference_dataset = config.get("reference_dataset")
            print("Using reference dataset from config.")
        if not reference_dataset:
            raise ValueError("Reference dataset must be provided either as input artifact or in config.")
        properties_ls = ip["properties"]
        task_name = ip["task_name"]
        work_dir = Path(task_name)
        # read structures (in xyz format) and properties (in npy format) into structure_summaries
        try:
            ase_atoms = ase.io.read(structures, ":")
            structures = [AseAtomsAdaptor.get_structure(x) for x in ase_atoms]
        except Exception as e:
            print(f"Failed to read structure files!: {e}")
            raise RuntimeError("Failed to read structure files!") from e
        properties = {k: np.load(v) for k, v in properties_ls.items()}
        if energies:
            energies = np.load(energies)
            assert len(energies) == len(structures), "The number of energies and structures do not match!"
        structure_summaries = get_metrics_structure_summaries(structures,energies,properties)
        # prepare referece dataset in ReferenceDataset format
        reference_dataset= LMDBGZSerializer().deserialize(reference_dataset)
        
        with set_directory(work_dir):
        # create metrics evaluator
            evaluator= MetricsEvaluator.from_structure_summaries(
                    structure_summaries,
                    reference_dataset,
                    property_constraints=config.get("property_constraints")      
                    )
            metrics= get_metrics(
                evaluator,
                metrics=config.get("metrics", []),
            )
            metric_results = evaluator.compute_metrics(
                metrics=metrics,
                calc_pre_aggregate=config.get("calc_pre_aggregate",True)
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
            mask= np.logical_and.reduce(pre_aggregate_masks)
            selected_structure_summaries = [s for s, m in zip(structure_summaries, mask) if m]
        
            # write the structure to xzy format
            try:
                selected_structures = [s.entry.structure for s in selected_structure_summaries]
                write("selected_structures.extxyz", [AseAtomsAdaptor.get_atoms(s) for s in selected_structures])
            except Exception as e:
                print(f"Failed to write structure files!: {e}")
                raise RuntimeError("Failed to write structure files!") from e
        
            selected_properties = {}
            for k in properties.keys():
                arr=np.array([s.properties[k] for s in selected_structure_summaries])
                np.save(f"{k}.npy",arr)
                selected_properties[k]=f"{k}.npy"
            
            # update reference
            update_reference = config.get("update_reference", False)
            if update_reference:
                #if not isinstance(reference_dataset, ReferenceDataset):
                reference_entries = [entry for entry in reference_dataset]
                reference_entries.extend([s.entry for s in selected_structure_summaries])
                reference_dataset = ReferenceDataset.from_entries(
                    "reference_dataset",
                    reference_entries
                    )
                LMDBGZSerializer().serialize(reference_dataset, "updated_reference_dataset")
                reference_dataset = work_dir / "updated_reference_dataset"
            else:
                reference_dataset = ip["reference_dataset"]
            
            # Prepare energy_above_hull output
            energy_above_hull_path = None
            if energy_above_hull_values is not None:
                energy_above_hull_path = work_dir / "energy_above_hull.npy"
            
        return OPIO({
            "selected_structures": work_dir / "selected_structures.extxyz",
            "selected_structures_properties": {k: work_dir / v for k, v in selected_properties.items()},
            "results": metric_results,
            "energy_above_hull": energy_above_hull_path,
            #"reference_dataset": reference_dataset,
        })
        
    
def get_metrics(
    evaluator: MetricsEvaluator,
    metrics: list[str]=[]):
    """Get the metric types from a list of metric names.

    Args:
        evaluator (MetricsEvaluator): The evaluator to use.
        metrics (list[str], optional): A list of name. Defaults to [].

    Returns:
        list[object]: _description_
    """
    if len(metrics) == 0:
        return "all"
    
    all_metrics = {metric.name:metric for metric in evaluator.available_metrics}
    
    metrics_list = []
    for k,v in all_metrics.items():
        if k in metrics:
            metrics_list.append(v)
    return metrics_list
