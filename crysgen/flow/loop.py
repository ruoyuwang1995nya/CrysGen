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

from crysgen.utils.step_config import init_executor

def _pop_executor(config: Dict) -> (Dict, Dict, Any):
    """Pop out executor related configs from step config."""
    config = deepcopy(config)
    template_config = config.pop("template_config", {})
    executor_config = config.pop("executor_config", {})
    executor = init_executor(executor_config)
    return config, template_config, executor


class CrysGenBlock(Steps):
    def __init__(
        self
    ):
        self._input_parameters = {
            "block_id": InputParameter(),
        }
        self._input_artifacts = {
            "init_model": InputArtifact(optional=True),
        }
        self._output_parameters = {
            "block_id": OutputParameter(),
        }
        self._output_artifacts = {
            "model": OutputArtifact(),
        }
        
        super().__init__(
            name="crys_gen_block",
            inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
            outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
        )
        
        self = _crys_gen_blk(self)
        
        
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
    
def _crys_gen_blk(
	steps: CrysGenBlock,
    name: str,
	train_generate_op: OPTemplate,
	evaluate_op: OPTemplate,
	upload_python_packages: Optional[List[os.PathLike]],
):

    ## super OP of train_generate
	train_generate = Step(
		name="generate",
		template=train_generate_op,
		parameters={
            "block_id": steps.inputs.parameters["block_id"],
		},
		artifacts={
			"model": steps.inputs.artifacts["init_model"],
			"seed_structures": steps.inputs.artifacts["seed_structures"],
		},
    key="--".join(["%s" % steps.inputs.parameters["block_id"], "train-generate"]),
	)
	steps.add(train_generate)


	evaluate = Step(
		name=f"evaluate",
		template=evaluate_op,
		parameters={
            "block_id": steps.inputs.parameters["block_id"],
			"evaluate_config": steps.inputs.parameters["evaluate_config"],
		},
		artifacts={
			"structures": current_structures,
			"model": steps.inputs.artifacts["init_model"],
		},
        key="--".join(["%s" % steps.inputs.parameters["block_id"], "evaluate"]),
		)
	steps.add(evaluate)
	current_structures = evaluate.outputs.artifacts.get("structures", current_structures)
	current_report = evaluate.outputs.artifacts.get("report", current_report)

	steps.outputs.artifacts["generated_structures"]._from = evaluate.outputs.artifacts["structures"]
	steps.outputs.artifacts["evaluation_report"]._from = evaluate.outputs.artifacts["report"]
	steps.outputs.artifacts["model"]._from = generate_train.outputs.artifacts["model"]

	return steps
    
class CrysGenLoop(Steps):
    def __init__(
        self,
        name: str,
        num_blocks: int,
    ):
        self._input_parameters = {
            "task_name": InputParameter(),
            "num_blocks": InputParameter(),
        }
        self._input_artifacts = {
            "init_model": InputArtifact(optional=True),
        }
        self._output_parameters = {
            
        }
        self._output_artifacts = {
            
        }
        super().__init__(
            name=name,
            inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
            outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
        )
        
        self = _crys_gen_loop(
            self,
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
    
    
def _crys_gen_loop(
	steps: CrysGenLoop,
	name: str,
	scheduler_op: Type[OP],
	crys_gen_blk_op: OPTemplate,
	scheduler_config: Dict,
	upload_python_packages: Optional[List[os.PathLike]],
):
	sched_conf, sched_template, sched_executor = _pop_executor(scheduler_config)

	scheduler = Step(
		name="schedule",
		template=_pop_executor(scheduler_op, sched_template, upload_python_packages),
		parameters={
			"task_name": steps.inputs.parameters["task_name"],
			"scheduler_config": steps.inputs.parameters["scheduler_config"],
		},
		artifacts={
			"init_model": steps.inputs.artifacts["init_model"],
			"prior_data": steps.inputs.artifacts["prior_data"],
		},
		executor=sched_executor,
		**sched_conf,
	)
	steps.add(scheduler)

	crys_gen_blk = Step(
        name=name + "-crys-gen",
        template=crys_gen_blk_op,
        parameters={
            "block_id": "iter-%s" % scheduler.outputs.parameters["iter_id"],
            "expl_tasks": scheduler.outputs.parameters["task_grp"],
            "conf_selector": steps.inputs.parameters["conf_selector"],
            "select_confs_config": steps.inputs.parameters["select_confs_config"],
            "template_script": steps.inputs.parameters["template_script"],
            "train_config": scheduler.outputs.parameters["train_config"],
            "explore_config": steps.inputs.parameters["explore_config"],
            "fp_config": steps.inputs.parameters["fp_config"],
            "evaluate_config": steps.inputs.parameters["evaluate_config"],
            "collect_data_config": steps.inputs.parameters["collect_data_config"],
        },
        artifacts={
            "expl_model": scheduler.outputs.artifacts[
                "expl_model"
            ],  # model for exploration
            "init_model": scheduler.outputs.artifacts[
                "init_model"
            ],  # starting point for finetune
            "init_data": steps.inputs.artifacts["init_data"],
            "iter_data": steps.inputs.artifacts["iter_data"],
        },
        key="--".join(
            ["iter-%s" % scheduler.outputs.parameters["iter_id"], "crys-gen-loop"]
        ),
        when="%s == false" % (scheduler.outputs.parameters["converged"]),
    )

	steps.add(crys_gen_blk)
	
	# parameters for the next iteration
	next_parameters = {
        "block_id": scheduler.outputs.parameters["next_iter_id"],
	}
 
	next_step = Step(
		name=name="-crys-gen-next",
		template=steps,
		parameters=next_parameters,
		artifacts={},
		when="%s == false" % (scheduler.outputs.parameters["converged"]),
		key="--".join(
      		["iter-%s" % scheduler.outputs.parameters["next_iter_id"], "crys-gen-loop"]
			)
	)
 
	steps.add(next_step)
	steps.outputs.artifacts["generated_structures"]._from = crys_gen_blk.outputs.artifacts["generated_structures"]
	steps.outputs.artifacts["evaluation_report"]._from = crys_gen_blk.outputs.artifacts["evaluation_report"]
	steps.outputs.artifacts["model"]._from = crys_gen_blk.outputs.artifacts["model"]
	return steps