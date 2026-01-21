from typing import Dict, Optional, Any,List
import os
from copy import deepcopy
from pathlib import Path
from dflow import (
    Step, 
    Steps,
    argo_len,
    argo_sequence,
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Steps,
    OPTemplate,
)
    
from dflow.python import (
    PythonOPTemplate,
    Slices,
    OP,
    Artifact,
    Parameter,
    BigParameter
)
from crysgen.utils.step_config import init_executor
from crysgen.op.relax import RelaxFF
from crysgen.op.select_frame import SelectFrame
from crysgen.op.evaluate import SUNEvaluate
from crysgen.op.fp import (
    PrepVasp,
    RunVasprun
)
from .op import (
    SelectFrameVaspSolidElectrolyte,
    IonMD,
    SelectFrameIonMD
    )

def _pop_executor(config: Dict):
    """Pop out executor related configs from step config."""
    config = deepcopy(config)
    template_config = config.pop("template_config", {})
    executor_config = config.pop("executor", None)
    executor = init_executor(executor_config)
    return config, template_config, executor

@OP.function
def StructureDistributor(
    prefix: str,
    structures: Artifact(Path),
    config: dict,
) -> {"structures_ls": Artifact(List[Path]), "idx_ls": Parameter(List[str])}:
    """Distribute structures into a list of structure files.

    Args:
        prefix (str): Prefix for naming output files.
        structures (Artifact(Path)): Path to the input structures file.
        num_frames (int): Number of structures to include in each output file. Default is 1.

    Returns:
        structures_ls (Artifact(List[Path])): List of paths to structure files.
        idx_ls (Parameter(List[str])): List of indices corresponding to the structure files.
    """
    from ase.io import read, write
    from pathlib import Path
    import os
    
    num_frames = config.get("batch_size", 1)
    atoms_list = read(structures, index=":")
    structures_ls = []
    idx_ls = []

    output_dir = Path("distributed_structures")
    os.makedirs(output_dir, exist_ok=True)

    # Group structures into chunks of num_frames
    for chunk_idx in range(0, len(atoms_list), num_frames):
        chunk = atoms_list[chunk_idx:chunk_idx + num_frames]
        structure_path = output_dir / f"structure_{chunk_idx:06d}.xyz"
        write(structure_path, chunk)
        structures_ls.append(structure_path)
        idx_ls.append(f"{prefix}_{chunk_idx:06d}")
    
    print(idx_ls)

    return {
        "structures_ls": structures_ls,
        "idx_ls": idx_ls,
    }

@OP.function
def StructureCollector(
    structure_files: Artifact(List[Path]),
    energy_files: Artifact(List[Path])
) -> {"structures": Artifact(Path), "energies": Artifact(Path)}:
    """Collect and merge multiple structure and energy files.

    Args:
        structure_files (Artifact(List[Path])): List of structure file paths from multiple relax steps.
        energy_files (Artifact(List[Path])): List of energy file paths from multiple relax steps.

    Returns:
        structures (Artifact(Path)): Merged structure file containing all relaxed structures.
        energies (Artifact(Path)): Merged energies file containing all corresponding energies.
    """
    from ase.io import read, write
    from pathlib import Path
    import numpy as np

    # Collect all structures
    all_structures = []
    for structure_file in structure_files:
        atoms_list = read(structure_file, index=":")
        if isinstance(atoms_list, list):
            all_structures.extend(atoms_list)
        else:
            all_structures.append(atoms_list)

    # Write merged structures
    merged_structures = Path("merged_structures.extxyz")
    write(merged_structures, all_structures)

    # Collect all energies
    all_energies = []
    for energy_file in energy_files:
        energies = np.load(energy_file)
        if energies.ndim == 0:
            all_energies.append(float(energies))
        else:
            all_energies.extend(energies.tolist())

    # Write merged energies
    merged_energies = Path("merged_energies.npy")
    np.save(merged_energies, np.array(all_energies))

    return {
        "structures": merged_structures,
        "energies": merged_energies,
    }


class SolidElectrolyteMatterGen(Steps):
    """
    Solid Electrolyte MatterGen flow for crystal structure generation.

    This flow orchestrates the training and generation of solid electrolyte
    crystal structures using machine learning models. It includes steps for 
    preparing initial configurations, training models, generating new structures,
    evaluating them, and collecting data for further training.

    Args:
        name (str): The name of the flow.
        crys_gen_loop_op (OPTemplate): The operation template for the crystal
            generation loop.
        upload_python_packages (Optional[List[os.PathLike]]): List of Python      
    """
    def __init__(
        self,
        name: str,
        *args,
        **kwargs
    ):
        self._input_parameters = {
            "name": InputParameter(type=str),
            "config": InputParameter(), # BigParameter, includes all eval settings
            "vasp_config": InputParameter(), # BigParameter, includes all eval settings
        }
        self._input_artifacts = {
            "structures": InputArtifact(optional=True),
            "reference_dataset": InputArtifact(optional=True),
            "model": InputArtifact(optional=True),
        }
        self._output_parameters = {
        }
        self._output_artifacts = {
            "structures": OutputArtifact(optional=True),
            "results": OutputArtifact(optional=True),
        }

        super().__init__(
            name=name,
            inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
            outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
        )
    
    @classmethod
    def build(cls, 
              name,
              *args,
              misc_step_config: Dict[str, Any], 
              sun_eval_step_config: Dict[str, Any],
              ff_step_config: Dict[str, Any],
              dft_step_config: Dict[str, Any],
              upload_python_packages: Optional[List[os.PathLike]] = None,
              **kwargs) -> Steps:
        steps = cls(name=name)
        ## Initial relaxation with force field
        ff_config, ff_template_config, ff_executor = _pop_executor(ff_step_config)
        ff_template_slice_config = ff_config.pop("template_slice_config", {})
        sun_eval_config, sun_eval_template_config, sun_eval_executor = _pop_executor(sun_eval_step_config)
        dft_config, dft_template_config, dft_executor = _pop_executor(dft_step_config)
        dft_template_slice_config = dft_config.pop("template_slice_config", {})
        misc_config, misc_template_config, misc_executor = _pop_executor(misc_step_config)
        
        # Pre-sampling screening
        pre_screen = Step(
            "pre-screening",
            template= PythonOPTemplate(
                SelectFrame,
                python_packages=upload_python_packages,
                **misc_template_config,
            ),
            parameters={
                "config": steps.inputs.parameters["config"]["init_screen"]
                },
            artifacts={
                "structures": steps.inputs.artifacts["structures"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "pre-screening"]),
            executor=misc_executor,
            **misc_config
        )
        steps.add(pre_screen)
        
        distribute_structures_relax = Step(
            "distribute-structures-relax",
            template= PythonOPTemplate(
                StructureDistributor,
                **misc_template_config,
                python_packages=upload_python_packages,
                
            ),
            parameters={
                "prefix":"ff_relax",
                "config": steps.inputs.parameters["config"]["relax_ff"]                
                },
            artifacts={"structures": pre_screen.outputs.artifacts["selected_structures"]},
            key="--".join(["%s"%steps.inputs.parameters["name"], "distribute-relax-structures"]),
            executor=misc_executor,
            **misc_config
        )
        steps.add(distribute_structures_relax)
        
        
        # Coarse relaxation
        relax = Step(
            "ff-relaxation",
            template= PythonOPTemplate(
                RelaxFF,
                slices=Slices(
                    '{{item}}',
                    input_parameter=["task_name"],
                    input_artifact=["structures"],
                    output_artifact=["relaxed_structures", "energies"],
                    group_size=1,
                    pool_size=1
                ),
                python_packages=upload_python_packages,
                **ff_template_config,
            ),
            parameters={
                "task_name": distribute_structures_relax.outputs.parameters["idx_ls"],
                "config": steps.inputs.parameters["config"]["relax_ff"]
                },
            artifacts={
                "structures":distribute_structures_relax.outputs.artifacts["structures_ls"],
                "model": steps.inputs.artifacts["model"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "ff-relax","{{item}}"]),
            with_sequence=argo_sequence(argo_len(distribute_structures_relax.outputs.parameters["idx_ls"])),
            executor=ff_executor,
            **ff_config
        )
        steps.add(relax)
        
        
        collect_relax = Step(
            "collect-relaxation",
            template= PythonOPTemplate(
                StructureCollector,
                python_packages=upload_python_packages,
                **misc_template_config,
            ),
            parameters={},
            artifacts={
                "structure_files": relax.outputs.artifacts["relaxed_structures"],
                "energy_files": relax.outputs.artifacts["energies"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "collect-relaxation"]),
            executor=misc_executor,
            **misc_config
        )
        steps.add(collect_relax)
        
        
        ## Inital S.U.N sampling
        sun_eval_ff = Step(
            "sun-eval-ff",
            template= PythonOPTemplate(
                SUNEvaluate,
                python_packages=upload_python_packages,
                **sun_eval_template_config,
            ),
            parameters={
                "config": steps.inputs.parameters["config"]["sun_eval"],
                "task_name": "sun_eval"
                },
            artifacts={
                "structures":collect_relax.outputs.artifacts["structures"],
                "reference_dataset": steps.inputs.artifacts["reference_dataset"],
                "energies": collect_relax.outputs.artifacts["energies"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "sun-eval-ff"]),
            executor=sun_eval_executor,
            **sun_eval_config
        )
        steps.add(sun_eval_ff)
        
        ## DFT calculations on selected structures
        prep_vasp = Step(
            "prep-vasp",
            template= PythonOPTemplate(
                PrepVasp,
                python_packages=upload_python_packages,
                **misc_template_config,
            ),
            parameters={
                "config": steps.inputs.parameters["vasp_config"],
                },
            artifacts={
                "confs":sun_eval_ff.outputs.artifacts["selected_structures"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "prep-vasp"]),
            executor=misc_executor,
            **misc_config
        )
        steps.add(prep_vasp)
               
        run_vasp = Step(
            "run-vasp",
            template=PythonOPTemplate(
                RunVasprun,
                slices=Slices(
                    "int('{{item}}')",
                    input_parameter=["task_name"],
                    input_artifact=["task_path"],
                    output_artifact=["log", "labeled_data", "extra_outputs"],
                    **dft_template_slice_config,
            ),
            python_packages=upload_python_packages,
            **dft_template_config,
        ),
            parameters={
                "task_name": prep_vasp.outputs.parameters["task_names"],
                "config": steps.inputs.parameters["vasp_config"],
                },
            artifacts={
                "task_path": prep_vasp.outputs.artifacts["task_paths"],
                #"model": prep_run_steps.inputs.artifacts["model"],
            },
            with_sequence=argo_sequence(
                argo_len(prep_vasp.outputs.parameters["task_names"]), format="%06d"
                ),
            key="--".join(["%s"%steps.inputs.parameters["name"], "run-vasp","-{{item}}"]),
            executor=dft_executor,
            **dft_config,
            )
        steps.add(run_vasp)
        
        ## DFT evaluation steps
        select_vasp= Step(
            "select-vasp",
            template= PythonOPTemplate(
                SelectFrameVaspSolidElectrolyte,
                python_packages=upload_python_packages,
                **sun_eval_template_config,
            ),
            parameters={},
            artifacts={
                "structures": run_vasp.outputs.artifacts["labeled_data"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "select-vasp"]),
            executor=sun_eval_executor,
            **sun_eval_config                                                                   
        )
        steps.add(select_vasp)
        
        # S.U.N selection based on VASP results
        sun_eval_vasp = Step(
            "sun-eval-vasp",
            template= PythonOPTemplate(
                SUNEvaluate,
                python_packages=upload_python_packages,
                **sun_eval_template_config,
            ),
            parameters={
                "config": steps.inputs.parameters["config"]["sun_eval"],
                "task_name": "sun_eval_vasp"
                },
            artifacts={
                "structures":select_vasp.outputs.artifacts["selected_structures"],
                "reference_dataset": steps.inputs.artifacts["reference_dataset"],
                "energies": select_vasp.outputs.artifacts["energies"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "sun-eval-vasp"]),
            executor=sun_eval_executor,
            **sun_eval_config
        )
        steps.add(sun_eval_vasp)
        
        ## simple OP for structure distribution
        distribute_structures = Step(
            "distribute-structures",
            template= PythonOPTemplate(
                StructureDistributor,
                **misc_template_config,
                python_packages=upload_python_packages,
                
            ),
            parameters={"prefix":"ion_md","config":{}},
            artifacts={"structures": sun_eval_vasp.outputs.artifacts["selected_structures"]},
            key="--".join(["%s"%steps.inputs.parameters["name"], "distribute-structures"]),
            executor=misc_executor,
            **misc_config
        )
        steps.add(distribute_structures)
        
        ## Ion transport simulations
        run_md = Step(
            name="ion-md",
            template = PythonOPTemplate(
                IonMD, 
                slices=Slices(
                    '{{item}}',
                    input_parameter=["task_name"],
                    input_artifact=["structure"],
                    output_artifact=["traj", "results"],
                    #output_parameter=["results"],
                    **ff_template_slice_config,
                ),
                python_packages=upload_python_packages,
                **ff_template_config,
                ),
            parameters={
                "task_name": distribute_structures.outputs.parameters["idx_ls"],
                "config": steps.inputs.parameters["config"]["ion_md"],
                },
            artifacts={
                "structure": distribute_structures.outputs.artifacts["structures_ls"],
                "model": steps.inputs.artifacts["model"],
            },
            with_sequence=argo_sequence(argo_len(distribute_structures.outputs.parameters["idx_ls"])),
            key="--".join(["%s"%steps.inputs.parameters["name"], "ion-md","{{item}}"]),
            executor=ff_executor,
            **ff_config,
        )        
        steps.add(run_md)
                
        select_ion_md= Step(
            "select-ion-md",
            template= PythonOPTemplate(
                SelectFrameIonMD,
                python_packages=upload_python_packages,
                **misc_template_config,
            ),
            parameters={
                #"results": run_md.outputs.parameters["results"],
                "config": steps.inputs.parameters["config"]["ion_eval"],
                },
            artifacts={
                "results": run_md.outputs.artifacts["results"],
                "structures": distribute_structures.outputs.artifacts["structures_ls"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "select-ion-md"]),
            executor=misc_executor,
            **misc_config
        )
        steps.add(select_ion_md)
        
        steps.outputs.artifacts["structures"]._from = select_ion_md.outputs.artifacts["selected_structures"]
        steps.outputs.artifacts["results"]._from = select_ion_md.outputs.artifacts["selected_results"]
        return steps