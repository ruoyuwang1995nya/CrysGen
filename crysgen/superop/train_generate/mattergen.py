
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Any,List, Union
import os
from copy import deepcopy
from dflow import (
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
    Artifact, 
    BigParameter, 
    OPIO, 
    OPIOSign, 
    OP,
    PythonOPTemplate
)
from dflow import Step, Steps
from crysgen.utils.step_config import init_executor
from crysgen.op.train import Train
from crysgen.op.generate import Generate
from .train_generate import TrainGeneration


def _pop_executor(config: Dict):
    """Pop out executor related configs from step config."""
    config = deepcopy(config)
    template_config = config.pop("template_config", {})
    executor_config = config.pop("executor", None)
    executor = init_executor(executor_config)
    return config, template_config, executor

class MatterGen(TrainGeneration):
    """
    MatterGen flow for crystal structure generation.

    This flow orchestrates the training and generation of crystal structures
    using machine learning models. It includes steps for preparing initial
    configurations, training models, generating new structures, evaluating
    them, and collecting data for further training.

    Args:
        name (str): The name of the flow.
        crys_gen_loop_op (OPTemplate): The operation template for the crystal
            generation loop.
        upload_python_packages (Optional[List[os.PathLike]]): List of Python      
    """
    
    @classmethod
    def build(cls, 
              name,
              *args,
              step_config: Dict[str, Any],
              upload_python_packages: Optional[List[os.PathLike]] = None,
              **kwargs) -> MatterGen:
        """Subclasses must add their Steps composing the evaluation logic."""
        # instantiate the SuperOP        
        steps=cls(name=name)
        
        step_config, step_template_config, step_executor = _pop_executor(
            step_config.get("train_step_config", {})
        )
        train=Step(
            name="mattergen-train",
            template = PythonOPTemplate(
                Train, 
                python_packages=upload_python_packages,
                **step_template_config
                ),
            parameters={
                "task_name":"mattergen-train",
                "config":steps.inputs.parameters["train_config"],
                },
            artifacts={
                "base_model":steps.inputs.artifacts["model"],
                "training_data":steps.inputs.artifacts["training_data"],
                "valid_data":steps.inputs.artifacts["valid_data"],
                "test_data":steps.inputs.artifacts["test_data"],
                },
            key="--".join(["%s"%steps.inputs.parameters["name"], "mattergen-train"]),
            executor=step_executor,
            **step_config
        )
        steps.add(train)
        
        generate=Step(
            name="generate",
            template = PythonOPTemplate(
                Generate, 
                python_packages=upload_python_packages,
                **step_template_config
                ),
            parameters={
                "task_name":"mattergen-generate",
                "config":steps.inputs.parameters["generate_config"],
                },
            artifacts={
                "model":train.outputs.artifacts["model"],
            },
            key="--".join(["%s"%steps.inputs.parameters["name"], "mattergen-generate"]),
            executor=step_executor,
            **step_config
        )
        steps.add(generate)
        steps.outputs.artifacts["generated_structures"]._from = generate.outputs.artifacts["generated_structures"]
        steps.outputs.artifacts["model"]._from = train.outputs.artifacts["model"]
        steps.outputs.parameters["results"].value_from_parameter = generate.outputs.parameters["results"]
        return steps
        
        