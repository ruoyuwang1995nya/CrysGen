"""Utility functions for downloading workflow artifacts."""

import logging
from pathlib import Path
from dflow import download_artifact


def download_step_artifacts(
    step,
    step_key: str,
    output_dir: Path,
    download_inputs: bool = True,
    download_outputs: bool = True,
):
    """Download artifacts from a single step.
    
    Args:
        step: The dflow step object
        step_key: The step key string
        output_dir: Base directory for downloads
        download_inputs: Whether to download input artifacts
        download_outputs: Whether to download output artifacts
    """
    step_dir = output_dir / step_key
    
    # Download inputs
    if download_inputs and hasattr(step, 'inputs') and hasattr(step.inputs, 'artifacts'):
        input_dir = step_dir / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        
        for artifact_name, artifact in step.inputs.artifacts.items():
            try:
                artifact_path = input_dir / artifact_name
                download_artifact(artifact, path=artifact_path, skip_exists=True)
                print(f"  Downloaded input: {artifact_name}")
            except (NotImplementedError, FileNotFoundError, Exception) as e:
                logging.warning(f"  Cannot download input artifact '{artifact_name}': {e}")
    
    # Download outputs
    if download_outputs and hasattr(step, 'outputs') and hasattr(step.outputs, 'artifacts'):
        output_dir_artifacts = step_dir / "outputs"
        output_dir_artifacts.mkdir(parents=True, exist_ok=True)
        
        for artifact_name, artifact in step.outputs.artifacts.items():
            try:
                artifact_path = output_dir_artifacts / artifact_name
                download_artifact(artifact, path=artifact_path, skip_exists=True)
                print(f"  Downloaded output: {artifact_name}")
            except (NotImplementedError, FileNotFoundError, Exception) as e:
                logging.warning(f"  Cannot download output artifact '{artifact_name}': {e}")
    
    print(f"Downloaded artifacts for step: {step_key}")
