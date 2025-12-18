"""Abstract evaluation super-step with unified I/O for crysgen flows."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

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


class Evaluation(Steps, ABC):
    """Base class for evaluation pipelines.

    Subclasses add concrete steps (e.g., relaxation + metrics) but share a
    unified interface:
    - inputs: ``task_name``, ``structures``, ``reference_dataset``, optional
      ``energies`` and ``properties``, plus ``config``.
    - outputs: ``results`` (structured dict), subclasses may add more.
    """

    def __init__(
        self,
        name: str,
        *,
        evaluate_op: Optional[OPTemplate] = None,
    ):
        self._input_parameters = {
            "name": InputParameter(type=str),
            "config": InputParameter(), # BigParameter, includes all eval settings
            #"properties": InputParameter(optional=True, value={}),
        }
        self._input_artifacts = {
            "structures": InputArtifact(Path,optional=True),
            "reference_dataset": InputArtifact(Path, optional=True),
            "model": InputArtifact(Path, optional=True),
            #"energies": InputArtifact(Path, optional=True),
        }
        self._output_parameters = {
            "results": OutputParameter(), # BigParameter(dict), all evaluation results
        }
        self._output_artifacts = {
            "structures": OutputArtifact(Path, optional=True),
        }

        super().__init__(
            name=name,
            inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
            outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
        )

        # Let subclasses build their internal steps using the provided templates.
        #self._build(evaluate_op=evaluate_op)

    @abstractmethod
    def build(self, *, evaluate_op: Optional[OPTemplate]) -> None:
        """Subclasses must add their Steps composing the evaluation logic."""

