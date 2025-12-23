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
    Parameter
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
) -> {"structures_ls": Artifact(List[Path]), "idx_ls": Parameter(List[str])}:
    """Distribute structures into a list of single-structure files.

    Args:
        structures (Artifact(Path)): Path to the input structures file.

    Returns:
        structures_ls (Artifact(List[Path])): List of paths to single-structure files.
        idx_ls (Parameter(List[int])): List of indices corresponding to the structures.
    """
    from ase.io import read, write
    from pathlib import Path
    import os

    atoms_list = read(structures, index=":")
    structures_ls = []
    idx_ls = []

    output_dir = Path("distributed_structures")
    os.makedirs(output_dir, exist_ok=True)

    for idx, atoms in enumerate(atoms_list):
        structure_path = output_dir / f"structure_{idx:06d}.xyz"
        write(structure_path, atoms)
        structures_ls.append(structure_path)
        idx_ls.append(f"{prefix}_{idx:06d}")

    return {
        "structures_ls": structures_ls,
        "idx_ls": idx_ls,
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
            "results": OutputParameter(), # BigParameter(dict), all evaluation results
        }
        self._output_artifacts = {
            "structures": OutputArtifact(optional=True),
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
        
        # Coarse relaxation
        relax = Step(
            "ff-relaxation",
            template= PythonOPTemplate(
                RelaxFF,
                python_packages=upload_python_packages,
                **ff_template_config,
            ),
            parameters={
                "task_name": "relax_ff",
                "config": steps.inputs.parameters["config"]["relax_ff"]
                },
            artifacts={
                "structures":pre_screen.outputs.artifacts["selected_structures"],
                "model": steps.inputs.artifacts["model"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "ff-relaxation"]),
            executor=ff_executor,
            **ff_config
        )
        steps.add(relax)
        
        ## Inital S.U.N sampling
        sun_eval_ff = Step(
            "sun-eval-ff",
            template= PythonOPTemplate(
                SUNEvaluate,
                python_packages=upload_python_packages,
                **ff_template_config,
            ),
            parameters={
                "config": steps.inputs.parameters["config"]["sun_eval"],
                "task_name": "sun_eval"
                },
            artifacts={
                "structures":relax.outputs.artifacts["relaxed_structures"],
                "reference_dataset": steps.inputs.artifacts["reference_dataset"],
                "energies": relax.outputs.artifacts["energies"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "sun-eval-ff"]),
            executor=ff_executor,
            **ff_config
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
                "structures":relax.outputs.artifacts["relaxed_structures"],
                "reference_dataset": steps.inputs.artifacts["reference_dataset"],
                "energies": relax.outputs.artifacts["energies"],
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
            parameters={"prefix":"ion_md"},
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
                    output_artifact=["traj"],
                    output_parameter=["results"],
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
                "results": run_md.outputs.parameters["results"],
                "config": steps.inputs.parameters["config"]["ion_eval"],
                },
            artifacts={
                #"results": run_md.outputs.artifacts["results"],
                "structures": distribute_structures.outputs.artifacts["structures_ls"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "select-ion-md"]),
            executor=misc_executor,
            **misc_config
        )
        steps.add(select_ion_md)
        
        steps.outputs.artifacts["structures"]._from = select_ion_md.outputs.artifacts["selected_structures"]
        steps.outputs.parameters["results"].value_from_parameter = select_ion_md.outputs.parameters["selected_results"]
        return steps