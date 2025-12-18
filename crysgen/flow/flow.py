"""Lightweight workflow scaffold for crysgen training/generation/evaluation.

This mirrors the high-level structure of the pfd-kit flows: a scheduler stage
drives one cycle of generate → evaluate (modular) → collect → train. Each stage
expects a pluggable ``dflow`` OP/PythonOP so you can reuse existing operators
and swap implementations without rewriting the flow.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Type

from dflow import (
	InputArtifact,
	InputParameter,
	Inputs,
	OutputArtifact,
	OutputParameter,
    OPTemplate,
	Outputs,
	Step,
	Steps,
)
from dflow.python import OP, PythonOPTemplate

from crysgen.utils.step_config import init_executor


def _python_op_template(
	op: Type[OP],
	template_config: Dict,
	upload_python_packages: Optional[List[os.PathLike]],
):
	"""Wrap an OP with PythonOPTemplate while passing through template configs."""

	return PythonOPTemplate(op, python_packages=upload_python_packages, **template_config)


class CrysGenFlow(Steps):
	"""Scheduler-driven crysgen workflow.

	Expected I/O contracts (you can adjust in your custom OPs):
	- scheduler_op outputs parameter ``stage_id``.
	- generate_op outputs artifact ``structures`` (e.g., extxyz) and optionally ``model``.
	- each evaluate_op takes ``structures`` and outputs ``report`` and/or filtered ``structures``.
	- collect_op takes ``structures`` (plus optional reports) and outputs ``collected_data``.
	- train_op takes ``collected_data`` (and optional ``init_model``) and outputs ``model``.
	"""

	def __init__(
		self,
		name: str,
		crys_gen_loop_op: OPTemplate,
	):
		self._input_parameters = {
			"task_name": InputParameter(),
			"scheduler_config": InputParameter(),
			"generate_config": InputParameter(),
			"evaluate_config": InputParameter(),
			"collect_config": InputParameter(),
			"train_config": InputParameter(),
		}
		self._input_artifacts = {
			"init_model": InputArtifact(optional=True),
			"seed_structures": InputArtifact(optional=True),
			"prior_data": InputArtifact(optional=True),
		}
		self._output_parameters = {
			"stage_id": OutputParameter(),
		}
		self._output_artifacts = {
			"model": OutputArtifact(),
			"generated_structures": OutputArtifact(),
			"collected_data": OutputArtifact(),
			"evaluation_report": OutputArtifact(optional=True),
		}

		super().__init__(
			name=name,
			inputs=Inputs(parameters=self._input_parameters, artifacts=self._input_artifacts),
			outputs=Outputs(parameters=self._output_parameters, artifacts=self._output_artifacts),
		)

		_build_flow(
			steps=self,
			name=name,
			crys_gen_loop_op=crys_gen_loop_op
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


def _build_flow(
	steps: CrysGenFlow,
    name:str,
	crys_gen_loop_op: OPTemplate,
):
	loop  = Step(
        name=name+"crys-gen-loop",
        template=crys_gen_loop_op,
        parameters={},
        artifacts={
            "init_model": steps.inputs.artifacts["init_model"],
            "seed_structures": steps.inputs.artifacts["seed_structures"],
            "prior_data": steps.inputs.artifacts["prior_data"],
        },
        key="crys-gen-loop",
	)
	steps.add(loop)
	steps.outputs.artifacts["model"]._from = loop.outputs.artifacts["model"]
	steps.outputs.artifacts["iter_data"]._from = loop.outputs.artifacts["iter_data"]
	return steps
