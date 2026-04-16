
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
    argo_len,
    argo_sequence,
    argo_range
)
from dflow.python import (
    Artifact, 
    BigParameter, 
    OPIO, 
    OPIOSign, 
    OP,
    Parameter,
    PythonOPTemplate,
    Slices
)
from dflow import Step, Steps
from crysgen.utils import set_directory,make_path
from crysgen.utils.step_config import init_executor
from crysgen.op.train import Train
from crysgen.op.generate import Generate
from .train_generate import TrainGeneration
from crysgen.tools.mattergen import mattergen_data
from ase.io import read
import logging  

def _pop_executor(config: Dict):
    """Pop out executor related configs from step config."""
    config = deepcopy(config)
    template_config = config.pop("template_config", {})
    executor_config = config.pop("executor", None)
    executor = init_executor(executor_config)
    return config, template_config, executor


class PrepareData(OP):
    """Prepare training data in mattergen format with optional train/val/test split."""
    
    def __init__(self):
        pass
    
    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign(
            {
                "task_name": Parameter(str),
                "config": Parameter(dict),
                "training_data": Artifact(List[Path]),
                "val_data": Artifact(List[Path], optional=True),
                "test_data": Artifact(List[Path], optional=True),
            },
        )
    
    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {
                "training_data": Artifact(Path),
                "val_data": Artifact(Path, optional=True),
                "test_data": Artifact(Path, optional=True),
            },
        )
    
    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        import numpy as np
        from ase.io import write
        
        task_name = ip["task_name"]
        config = ip["config"]
        training_data = ip["training_data"]
        val_data = ip.get("val_data")
        print(val_data)
        test_data = ip.get("test_data")
        
        results = {}
        workdir = Path(make_path(task_name))
        
        with set_directory(workdir):
            # Read all training structures
            train_atoms_ls = []
            for path in training_data:
                train_atoms_ls.extend(read(path.as_posix(), ':'))
            
            # Check if we need to split the training data
            val_ratio = config.get("val_ratio", 0.0)
            test_ratio = config.get("test_ratio", 0.0)
            
            if (val_data is None and val_ratio > 0) or (test_data is None and test_ratio > 0):
                logging.info("Splitting training data into train/val/test sets.")
                # Perform random split
                n_total = len(train_atoms_ls)
                indices = np.random.permutation(n_total)
                
                n_test = int(n_total * test_ratio) if test_data is None else 0
                n_val = int(n_total * val_ratio) if val_data is None else 0
                n_train = n_total - n_val - n_test
                
                train_indices = indices[:n_train]
                val_indices = indices[n_train:n_train + n_val]
                test_indices = indices[n_train + n_val:]
                
                # Split the data
                final_train_atoms = [train_atoms_ls[i] for i in train_indices]
                
                # Save training data
                write("train_tmp.extxyz", final_train_atoms)
                mattergen_data(
                    ase_extxyz_file="train_tmp.extxyz",
                    mattergen_data=Path("training_data"),
                    properties=config.get("properties", [])
                )
                results["training_data"] = workdir / "training_data"
                
                # Handle validation data
                if val_data is None and n_val > 0:
                    val_atoms_split = [train_atoms_ls[i] for i in val_indices]
                    write("val_tmp.extxyz", val_atoms_split)
                    mattergen_data(
                        ase_extxyz_file="val_tmp.extxyz",
                        mattergen_data=Path("val_data"),
                        properties=config.get("properties", [])
                    )
                    results["val_data"] = workdir / "val_data"
                elif val_data:
                    val_atoms_ls = []
                    for path in val_data:
                        val_atoms_ls.extend(read(path.as_posix(), ':'))
                    write("val_tmp.extxyz", val_atoms_ls)
                    mattergen_data(
                        ase_extxyz_file="val_tmp.extxyz",
                        mattergen_data=Path("val_data"),
                        properties=config.get("properties", [])
                    )
                    results["val_data"] = workdir / "val_data"
                
                # Handle test data
                if test_data is None and n_test > 0:
                    test_atoms_split = [train_atoms_ls[i] for i in test_indices]
                    write("test_tmp.extxyz", test_atoms_split)
                    mattergen_data(
                        ase_extxyz_file="test_tmp.extxyz",
                        mattergen_data=Path("test_data"),
                        properties=config.get("properties", [])
                    )
                    results["test_data"] = workdir / "test_data"
                elif test_data:
                    test_atoms_ls = []
                    for path in test_data:
                        test_atoms_ls.extend(read(path.as_posix(), ':'))
                    write("test_tmp.extxyz", test_atoms_ls)
                    mattergen_data(
                        ase_extxyz_file="test_tmp.extxyz",
                        mattergen_data=Path("test_data"),
                        properties=config.get("properties", [])
                    )
                    results["test_data"] = workdir / "test_data"
            else:
                # No split needed, use data as-is
                write("train_tmp.extxyz", train_atoms_ls)
                mattergen_data(
                    ase_extxyz_file="train_tmp.extxyz",
                    mattergen_data=Path("training_data"),
                    properties=config.get("properties", [])
                )
                results["training_data"] = workdir / "training_data"
                
                if val_data:
                    val_atoms_ls = []
                    for path in val_data:
                        val_atoms_ls.extend(read(path.as_posix(), ':'))
                    write("val_tmp.extxyz", val_atoms_ls)
                    mattergen_data(
                        ase_extxyz_file="val_tmp.extxyz",
                        mattergen_data=Path("val_data"),
                        properties=config.get("properties", [])
                    )
                    results["val_data"] = workdir / "val_data"
                
                if test_data:
                    test_atoms_ls = []
                    for path in test_data:
                        test_atoms_ls.extend(read(path.as_posix(), ':'))
                    write("test_tmp.extxyz", test_atoms_ls)
                    mattergen_data(
                        ase_extxyz_file="test_tmp.extxyz",
                        mattergen_data=Path("test_data"),
                        properties=config.get("properties", [])
                    )
                    results["test_data"] = workdir / "test_data"
        
        return OPIO(results)


class CollectGenerated(OP):
    """Merge generated structure files from sliced generate steps."""

    @classmethod
    def get_input_sign(cls) -> OPIOSign:
        return OPIOSign(
            {
                "structures_list": Artifact(List[Path]),
            }
        )

    @classmethod
    def get_output_sign(cls) -> OPIOSign:
        return OPIOSign(
            {
                "merged_structures": Artifact(Path),
            }
        )

    @OP.exec_sign_check
    def execute(self, ip: OPIO) -> OPIO:
        from ase.io import read, write

        structures_list = ip.get("structures_list", [])
        atoms_all = []
        for path in structures_list:
            atoms = read(path, ":")
            if isinstance(atoms, list):
                atoms_all.extend(atoms)
            else:
                atoms_all.append(atoms)

        merged_path = Path("merged_generated_structures.extxyz")
        if atoms_all:
            write(merged_path, atoms_all, format="extxyz")
        else:
            merged_path.write_text("")

        return OPIO({"merged_structures": merged_path})

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
              **kwargs
              ) -> MatterGen:
        """Subclasses must add their Steps composing the evaluation logic."""
        # instantiate the SuperOP        
        steps=cls(name=name)
        
        step_config, step_template_config, step_executor = _pop_executor(
            step_config)
        step_template_slice_config = step_config.pop("template_slice_config", {})
        prepare_data=Step(
            name='mattergen-prepare-data',
            template= PythonOPTemplate(
                PrepareData,
                **step_template_config,
                python_packages=upload_python_packages),
            parameters={
                "task_name": steps.inputs.parameters["name"],
                "config": steps.inputs.parameters["data_config"],
            },
            artifacts={
                "training_data": steps.inputs.artifacts["training_data"],
                "val_data": steps.inputs.artifacts["valid_data"],
                "test_data": steps.inputs.artifacts["test_data"],
            },
            key="--".join(["%s"%steps.inputs.parameters["name"], "mattergen-prepare-data"]),
            executor=step_executor,
            **step_config
        )
        steps.add(prepare_data)
        
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
                "training_data":prepare_data.outputs.artifacts["training_data"],
                "valid_data":prepare_data.outputs.artifacts["val_data"],
                "test_data":prepare_data.outputs.artifacts["test_data"],
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
                slices=Slices(
                    '{{item}}',
                    input_parameter=["task_name"],
                    output_parameter=["results"],
                    output_artifact=["generated_structures","extra_outputs"],
                    group_size=1,
                    pool_size=1
                ),
                python_packages=upload_python_packages,
                **step_template_config
                ),
            parameters={
                "task_name": train.outputs.parameters["generation_idx"],
                "config":steps.inputs.parameters["generate_config"],
                },
            artifacts={
                "model":train.outputs.artifacts["model"],
            },
            with_sequence=argo_sequence(
                argo_len(train.outputs.parameters["generation_idx"])#,format="%03d"
                ),
            key="--".join(["%s"%steps.inputs.parameters["name"], "mattergen-generate","{{item}}"]),
            executor=step_executor,
            **step_config
        )
        steps.add(generate)
        
        collect_generate = Step(
            name="collect-generate",
            template=PythonOPTemplate(
                CollectGenerated,
                python_packages=upload_python_packages,
                **step_template_config,
            ),
            parameters={},
            artifacts={
                "structures_list": generate.outputs.artifacts["generated_structures"],
            },
            key="--".join(["%s"%steps.inputs.parameters["name"], "collect-generate"]),
            executor=step_executor,
            **step_config,
        )
        steps.add(collect_generate)
        steps.outputs.artifacts["generated_structures"]._from = collect_generate.outputs.artifacts["merged_structures"]
        steps.outputs.artifacts["model"]._from = train.outputs.artifacts["model"]
        #steps.outputs.parameters["results"].value_from_parameter = generate.outputs.parameters["results"]
        return steps
        
        