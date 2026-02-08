import os
from copy import (
    deepcopy,
)
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Type,
    Union,
)

from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    Outputs,
    OPTemplate,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    if_expression,
)
from dflow.python import (
    OP,
    PythonOPTemplate,
)

from crysgen.superop.train_generate import TrainGeneration
from crysgen.superop.evaluate.solid_electrolyte import SolidElectrolyteMatterGen
from crysgen.op.schedule import Schedule
from crysgen.utils.step_config import init_executor

def _pop_executor(config: Dict) -> (Dict, Dict, Any):
    """Pop out executor related configs from step config."""
    config = deepcopy(config)
    template_config = config.pop("template_config", {})
    executor_config = config.pop("executor", None)
    executor = init_executor(executor_config)
    return config, template_config, executor
    
class SolidElectrolyteGenScreen(Steps):
    def __init__(
        self,
        name: str,
        num_blocks: int,
    ):
        self._input_parameters = {
            "task_name": InputParameter(),
            "iter_id": InputParameter(value=0),
            "iter_num": InputParameter(value=num_blocks),
            "generation_config": InputParameter(),
            "screening_config": InputParameter(),
            "scheduler_config": InputParameter(value={}),
            "vasp_config": InputParameter(),
        }
        self._input_artifacts = {
            "generative_model": InputArtifact(optional=True),
            "ff_model": InputArtifact(optional=True),
            "training_data": InputArtifact(),
            "valid_data": InputArtifact(optional=True),
            "test_data": InputArtifact(optional=True),
            "reference_dataset": InputArtifact(optional=True),
        }
        self._output_parameters = {
            
        }
        self._output_artifacts = {
            "generated_structures": OutputArtifact(optional=True),
            "model": OutputArtifact(optional=True),
        }
        super().__init__(
            name=name,
            inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
            outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
        )
        
    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts
    
    @classmethod
    def build(
        cls,
        name: str,
        num_blocks: int,
        train_generate_op: OPTemplate,
        screening_op: OPTemplate,
        scheduler_config: Dict,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ) -> "SolidElectrolyteGenScreen":
        """Build the SolidElectrolyteGenScreen workflow.
        
        Args:
            name: Name of the workflow
            num_blocks: Number of iterations
            train_generate_op: Operation template for training and generation
            screening_op: Operation template for screening
            scheduler_config: Configuration for the scheduler
            upload_python_packages: List of Python packages to upload
            
        Returns:
            Configured SolidElectrolyteGenScreen instance
        """
        # Create instance
        steps = cls(name=name, num_blocks=num_blocks)
        
        sched_conf, sched_template, sched_executor = _pop_executor(scheduler_config)

        train_generator = Step(
            name=name + "-train-generate",
            template=train_generate_op,
            parameters={
                "name": "iter-%s" % steps.inputs.parameters["iter_id"],
                "train_config": steps.inputs.parameters["generation_config"]["train_config"],
                "generate_config": steps.inputs.parameters["generation_config"]["generate_config"],
                "data_config": steps.inputs.parameters["generation_config"]["data_config"],
            },
            artifacts={
                "model": steps.inputs.artifacts["generative_model"],
                "training_data": steps.inputs.artifacts["training_data"],
                "valid_data": steps.inputs.artifacts["valid_data"],
                "test_data": steps.inputs.artifacts["test_data"],
            },
            key="--".join(
                ["iter-%s-%s" %(steps.inputs.parameters["iter_id"], "gen-screen-loop")]
            ),
            #when="%s == false" % (scheduler.outputs.parameters["converged"]),
        )

        steps.add(train_generator)
        
        screening = Step(
            name=name + "-screening",
            template=screening_op,
            parameters={
                "name": "iter-%s" % steps.inputs.parameters["iter_id"],
                "config": steps.inputs.parameters["screening_config"],
                "vasp_config": steps.inputs.parameters["vasp_config"],
            },
            artifacts={
                "structures": train_generator.outputs.artifacts["generated_structures"],
                "reference_dataset": steps.inputs.artifacts["reference_dataset"],
                "model": steps.inputs.artifacts["ff_model"],
            },
            key="--".join(
                ["iter-%s" % steps.inputs.parameters["iter_id"], "crys-gen-screening"]
            ),
        )
        steps.add(screening)
        
        scheduler = Step(
            name="schedule",
            template=PythonOPTemplate(
                Schedule,
                python_packages=upload_python_packages,
                **sched_template,
            ),
            parameters={
                "iter_id": steps.inputs.parameters["iter_id"],
                "config": steps.inputs.parameters["scheduler_config"],
                "iter_num": steps.inputs.parameters["iter_num"]
                
            },
            artifacts={
                "iter_data": steps.inputs.artifacts["training_data"],
                "incoming_data": screening.outputs.artifacts["structures"],
            },
            executor=sched_executor,
            **sched_conf,
        )
        steps.add(scheduler)

        # parameters for the next iteration
        next_parameters = {
            "task_name": steps.inputs.parameters["task_name"],
            "iter_id": scheduler.outputs.parameters["iter_id"],
            "generation_config": steps.inputs.parameters["generation_config"],
            "screening_config": steps.inputs.parameters["screening_config"],
            "scheduler_config": steps.inputs.parameters["scheduler_config"],
            "vasp_config": steps.inputs.parameters["vasp_config"],
            "iter_num": steps.inputs.parameters["iter_num"],
        }
     
        next_artifacts = {
            "generative_model": train_generator.outputs.artifacts["model"],
            "ff_model": steps.inputs.artifacts["ff_model"],
            "training_data": scheduler.outputs.artifacts["iter_data"],
            "valid_data": steps.inputs.artifacts["valid_data"],
            "test_data": steps.inputs.artifacts["test_data"],
            "reference_dataset": steps.inputs.artifacts["reference_dataset"],
        }
     
        next_step = Step(
            name=name + "-crys-gen-next",
            template=steps,
            parameters=next_parameters,
            artifacts=next_artifacts,
            when="%s == false" % (scheduler.outputs.parameters["stop"]),
            key="--".join(
                ["iter-%s"%scheduler.outputs.parameters["iter_id"], "gen-screen-loop"]
            )
        )
        steps.add(next_step)
        steps.outputs.artifacts["generated_structures"].from_expression = if_expression(
            _if=scheduler.outputs.parameters["stop"],
            _then=screening.outputs.artifacts["structures"],
            _else=next_step.outputs.artifacts["generated_structures"],
        )
        steps.outputs.artifacts["model"].from_expression = if_expression(
            _if=scheduler.outputs.parameters["stop"],
            _then=train_generator.outputs.artifacts["model"],
            _else=next_step.outputs.artifacts["model"],
        )
        return steps