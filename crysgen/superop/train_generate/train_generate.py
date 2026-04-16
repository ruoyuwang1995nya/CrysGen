"""Abstract evaluation super-step with unified I/O for crysgen flows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional,List

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
from dflow.python import Artifact, BigParameter, OPIO, OPIOSign, OP


class TrainGeneration(Steps, ABC):
    """Base class for training-generation pipelines.

    Subclasses add concrete steps (e.g., relaxation + metrics) but share a
    unified interface:
    - inputs: ``task_name``, ``structures``, ``reference_dataset``, optional
      ``energies`` and ``properties``, plus ``config``.
    - outputs: ``results`` (structured dict), subclasses may add more.
    """

    def __init__(
        self,
        name: str,
        *args,
        **kwargs
        #evaluate_op: Optional[OPTemplate] = None,
    ):
        self._input_parameters = {
            "name": InputParameter(type=str), # Block id
            "train_config": InputParameter(), # BigParameter, includes all training settings
            "generate_config": InputParameter(), # BigParameter, includes all generation settings
            "data_config": InputParameter(), # BigParameter, includes all data preparation settings
        }
        self._input_artifacts = {
            #"generated_structures": InputArtifact(Path), # path to generated structures
            "model": InputArtifact(), # path to pretrained model
            "training_data": InputArtifact(), # path to training data
            "valid_data": InputArtifact(optional=True), # path to validation data
            "test_data": InputArtifact(optional=True), # path to test data
        }
        self._output_parameters = {
            #"results": OutputParameter(value={}), # BigParameter(dict), all evaluation results
        }
        self._output_artifacts = {
            "generated_structures": OutputArtifact(optional=True),
            "model": OutputArtifact(optional=True)
        }

        super().__init__(
            name=name,
            inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
            outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
        )

        # Let subclasses build their internal steps using the provided templates.
        #self._build(evaluate_op=evaluate_op)

    @classmethod
    @abstractmethod
    def build(
        cls,
        *,
        name: str,
        step_config: Dict,
        upload_python_packages: Optional[list] = None,
        evaluate_op: Optional[OPTemplate] = None,
    ) -> "TrainGeneration":
        """Factory method to construct a concrete TrainGeneration Steps."""

