#!/usr/bin/env python
"""Entry point for solid electrolyte generation and screening workflow."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
from dflow import Workflow, Step
import logging
import re
from crysgen.entrypoint.common import global_config_workflow
from crysgen.flow.solid_electrolyte.loop import SolidElectrolyteGenScreen
from crysgen.superop.train_generate.mattergen import MatterGen
from crysgen.superop.evaluate.solid_electrolyte import SolidElectrolyteMatterGen
from crysgen.op.fp.vasp_input import VaspInputs
from crysgen.utils.workflow_query import print_steps, parse_index_string, get_steps_by_indices, get_steps_by_indices, sort_slice_ops, successful_step_keys
from crysgen.utils.artifacts import upload_artifact_and_print_uri, get_artifact_from_uri
from crysgen.utils.download_artifacts import download_step_artifacts

def get_superop(key):
    """[From DPGEN2] Get the super operation key for a given step key.

    Args:
        key (str): The step key.

    Returns:
        str: The super operation key, or None if not found.
    """
    if "prep-vasp" in key:
        return key.replace("prep-vasp", "prep-run-vasp")
    elif "run-vasp-" in key:
        return re.sub("run-vasp-[0-9]*", "prep-run-vasp", key)
    return None

def fold_keys(all_step_keys):
    """[From DPGEN2] Fold step keys by their super operation.
    
    Args:
        all_step_keys (List[str]): List of all step keys.
    Returns:
        dict: A dictionary mapping super operation keys to their folded step keys, e.g., {"prep-run-vasp":["run-vasp-0000"]}.
    """
    folded_keys = {}
    for key in all_step_keys:
        is_superop = False
        for superop in [ "prep-run-vasp"]:
            if superop in key:
                if key not in folded_keys:
                    folded_keys[key] = []
                is_superop = True
                break
        if is_superop:
            continue
        superop = get_superop(key)
        # if its super OP is succeeded, fold it into its super OP
        if superop is not None and superop in all_step_keys:
            if superop not in folded_keys:
                folded_keys[superop] = []
            folded_keys[superop].append(key)
        else:
            folded_keys[key] = [key]
    for k, v in folded_keys.items():
        if v == []:
            folded_keys[k] = [k]
    return folded_keys

def get_resubmit_keys(wf, unsuccessful_step_keys: bool = False):
    """[From DPGEN2] Get the keys of all steps in the workflow for resubmission.
    """
    all_step_keys = successful_step_keys(wf, unsuccessful_step_keys)
    all_step_keys = sort_slice_ops(
        all_step_keys,
        ["run-vasp","ion-md"],
    )
    
    return all_step_keys

def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_python_packages(module_names: List[str]) -> List[str]:
    """Setup Python packages to upload.
    
    Args:
        module_names: List of module names to check and upload
        
    Returns:
        List of package paths to upload
    """
    import importlib
    
    upload_python_packages = []
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, '__path__'):
                upload_python_packages.extend(list(module.__path__))
            else:
                print(f"Warning: Module '{module_name}' has no __path__ attribute")
        except ImportError:
            print(f"Warning: Module '{module_name}' not found in current environment")
    
    return upload_python_packages


def build_step_configs(config: dict) -> tuple:
    """Build step configurations from config dict."""
    default_step_config = config.get("default", {
        "executor": None,
        "template_config": {},
    })
    #print(default_step_config)
    train_step_config = config.get("train", default_step_config)
    vasp_step_config = config.get("fp", default_step_config)
    ion_md_step_config = config.get("ion_md", default_step_config)
    sun_eval_step_config = config.get("sun_eval", default_step_config)
    scheduler_step_config = config.get("scheduler", default_step_config)
    
    return (default_step_config, train_step_config, 
             vasp_step_config, ion_md_step_config,
            sun_eval_step_config, scheduler_step_config)


def build_vasp_config(config: dict) -> dict:
    """Build VASP configuration from config dict."""
    vasp_inputs = VaspInputs(**config["inputs"])
    
    vasp_config = {
        "inputs": vasp_inputs,
        "run": config.get("run", {}),
        "extra_output_files": config.get("extra_output_files", [])
    }
    return vasp_config


def build_artifacts(config: dict) -> dict:
    """Build artifacts from config dict."""
    artifacts = {
        #"training_data": upload_artifact(config["training_data"]),
    }
    if training_data := config.get("training_data"):
        try:
            artifacts["training_data"] = get_artifact_from_uri(training_data)
        except ValueError:
            artifacts["training_data"] = upload_artifact_and_print_uri(training_data, "training_data")
    else:
        raise ValueError("training_data artifact must be provided in config.")
    
    
    # Optional artifacts
    if generative_model := config.get("generative_model"):
        try:
            artifacts["generative_model"] = get_artifact_from_uri(generative_model)
        except ValueError:
            artifacts["generative_model"] = upload_artifact_and_print_uri(generative_model, "generative_model")
    
    if ff_model := config.get("ff_model"):
        try:
            artifacts["ff_model"] = get_artifact_from_uri(ff_model)
        except ValueError:
            artifacts["ff_model"] = upload_artifact_and_print_uri(ff_model, "ff_model")
    
    if valid_data := config.get("valid_data"):
        try:
            artifacts["valid_data"] = get_artifact_from_uri(valid_data)
        except ValueError:
            artifacts["valid_data"] = upload_artifact_and_print_uri(valid_data, "valid_data")
    
    if test_data := config.get("test_data"):
        try:
            artifacts["test_data"] = get_artifact_from_uri(test_data)
        except ValueError:
            artifacts["test_data"] = upload_artifact_and_print_uri(test_data, "test_data")
    
    if reference_dataset := config.get("reference_dataset"):
        try:
            artifacts["reference_dataset"] = get_artifact_from_uri(reference_dataset)
        except ValueError:  
            artifacts["reference_dataset"] = upload_artifact_and_print_uri(reference_dataset, "reference_dataset")
 
    return artifacts


def build_workflow(config: dict) -> Workflow:
    """Build workflow from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Workflow object (not yet submitted)
    """
    # Setup Python packages
    module_names = config.get("upload_python_packages", [])
    upload_python_packages = setup_python_packages(module_names)
    
    # Build step configs
    (default_step_config, train_step_config,  vasp_step_config, ion_md_step_config,sun_eval_step_config,
     scheduler_step_config) = build_step_configs(config["step_config"])

    # Get job configs
    workflow_config = config["jobs"].copy()
    
    # Build VASP config
    vasp_config = workflow_config["screening"].get("vasp", {})
    vasp_config = build_vasp_config(vasp_config)
    
    # Build artifacts
    artifacts_config = config["artifacts"].copy()
    artifacts = build_artifacts(artifacts_config)
    # Build SuperOPs
    train_generate_op = MatterGen.build(
        name="train-generate",
        step_config=train_step_config,
        upload_python_packages=upload_python_packages,
    )
    screening_op = SolidElectrolyteMatterGen.build(
        name="screening",
        misc_step_config=default_step_config,
        sun_eval_step_config=sun_eval_step_config,
        ff_step_config=ion_md_step_config,
        dft_step_config=vasp_step_config,
        upload_python_packages=upload_python_packages,
    )
    
    # Build Loop Steps
    num_blocks = config.get("num_iterations", 1)
    
    gen_screen_loop = SolidElectrolyteGenScreen.build(
        name=config.get("loop_name", "gen-screen-loop"),
        num_blocks=num_blocks,
        train_generate_op=train_generate_op,
        screening_op=screening_op,
        scheduler_config=scheduler_step_config,
        upload_python_packages=upload_python_packages,
    )
    
    # Create workflow
    wf_name = config.get("name", "solid-electrolyte-gen-screen")
    wf = Workflow(name=wf_name)
    
    loop_step = Step(
        name=config.get("step_name", "gen-screen-loop"),
        template=gen_screen_loop,
        parameters={
            "task_name": config.get("task_name", "gen-screen"),
            "iter_id": config.get("start_iter", 0),
            "iter_num": num_blocks,
            "generation_config": workflow_config.get("generation", {}),
            "screening_config": workflow_config.get("screening", {}),
            "scheduler_config": workflow_config.get("scheduler", {}),
            "vasp_config": vasp_config,
        },
        artifacts=artifacts
    )
    
    wf.add(loop_step)
    
    return wf


def submit_workflow(args):
    """Submit a new workflow."""
    config = load_config(args.config)
    
    # Setup dflow mode
    #dflow.config["mode"] = config.get("dflow_mode", "debug")
    print("Setting up global configuration...")
    global_config_workflow(config)
    
    # Build workflow
    print("Building workflow...")
    wf = build_workflow(config)
    
    #print(wf.to_dict())
    print(f"\nSubmitting workflow '{wf.name}' ...")
    wf.submit()
    
    print(f"Workflow submitted: {wf.id}")
    
    # Save workflow ID
    if args.save_wf_id:
        with open(args.save_wf_id, 'w') as f:
            f.write(wf.id)
        print(f"Workflow ID saved to {args.save_wf_id}")


def resubmit_workflow(args):
    """Resubmit workflow with selective step reuse."""
    config = load_config(args.config)
    
    # Setup dflow mode
    #dflow.config["mode"] = config.get("dflow_mode", "debug")
    global_config_workflow(config)
    
    # Get old workflow
    wf_old = Workflow(id=args.workflow_id)
    
    print(args.reuse_all_steps)
    # Get all reusable steps
    all_step_keys = get_resubmit_keys(
        wf_old,
        unsuccessful_step_keys=args.reuse_all_steps
    )
    
    folded_keys = fold_keys(all_step_keys)
    
    # If list_steps flag is set, print steps and exit
    if args.list_steps:
        print("\nReusable steps:")
        print_steps(all_step_keys)
        return
    
    # Select steps by indices if provided
    if args.step_indices:
        step_keys = get_steps_by_indices(all_step_keys, args.step_indices)
        print(f"Selected {len(step_keys)} steps by indices")
    else:
        step_keys = all_step_keys
        print(f"Reusing all {len(step_keys)} steps")
    
    if args.fold:
        reused_folded_keys = {}
        for key in step_keys:
            superop = get_superop(key)
            if superop is not None:
                if superop not in reused_folded_keys:
                    reused_folded_keys[superop] = []
                reused_folded_keys[superop].append(key)
            else:
                reused_folded_keys[key] = [key]
        for k, v in reused_folded_keys.items():
            # reuse the super OP iif all steps within it are reused
            if v != [k] and k in folded_keys and set(v) == set(folded_keys[k]):
                reused_folded_keys[k] = [k]
        step_keys = sum(reused_folded_keys.values(), [])
    
    if args.verbose:
        print("\nSteps to reuse:")
        print_steps(step_keys)
    
    # Build workflow
    wf = build_workflow(config)
    
    # Submit workflow with step reuse
    reuse_step = wf_old.query_step(key=step_keys)
    print(f"\nSubmitting workflow with {len(step_keys)} reused steps...")
    wf.submit(reuse_step=reuse_step)
    
    print(f"Workflow submitted: {wf.id}")
    
    # Save workflow ID
    if args.save_wf_id:
        with open(args.save_wf_id, 'w') as f:
            f.write(wf.id)
        print(f"Workflow ID saved to {args.save_wf_id}")


def download_workflow(args):
    """Download artifacts from workflow steps."""
    config = load_config(args.config)
    
    # Setup dflow mode
    #dflow.config["mode"] = config.get("dflow_mode", "debug")
    global_config_workflow(config)
    
    # Get workflow
    wf = Workflow(id=args.workflow_id)
    
    # Get all available steps
    all_step_keys = get_resubmit_keys(
        wf,
        unsuccessful_step_keys=args.include_failed
    )
    
    print(f"Found {len(all_step_keys)} available steps in workflow {args.workflow_id}")
    
    # If list_steps flag is set, print steps and exit
    if args.list_steps:
        print("\nAvailable steps:")
        print_steps(all_step_keys)
        return
    
    # Select steps by indices if provided
    if args.step_indices:
        step_keys = get_steps_by_indices(all_step_keys, args.step_indices)
        print(f"Selected {len(step_keys)} steps to download")
    else:
        step_keys = all_step_keys
        print(f"Downloading all {len(step_keys)} steps")
    
    if args.verbose:
        print("\nSteps to download:")
        print_steps(step_keys)
    
    # Set output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Query and download steps
    print(f"\nDownloading to: {output_dir.absolute()}")
    
    for step_key in step_keys:
        print(f"\nProcessing step: {step_key}")
        steps = wf.query_step(key=step_key)
        
        if len(steps) == 0:
            logging.warning(f"  Step '{step_key}' not found, skipping")
            continue
        
        step = steps[0]
        
        # Check if step succeeded (unless include_failed is True)
        if not args.include_failed and step.phase != "Succeeded":
            logging.warning(f"  Step '{step_key}' status is {step.phase}, skipping")
            continue
        
        # Download artifacts
        try:
            download_step_artifacts(
                step,
                step_key,
                output_dir,
                download_inputs=args.inputs,
                download_outputs=args.outputs,
            )
        except Exception as e:
            logging.error(f"  Failed to download step '{step_key}': {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    print(f"\nDownload complete. Artifacts saved to: {output_dir.absolute()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Solid Electrolyte Generation and Screening Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit new workflow
  %(prog)s submit -c config.json
  
  # List reusable steps
  %(prog)s resubmit wf-xxxxx -c config.json -l
  
  # Resubmit with all steps
  %(prog)s resubmit wf-xxxxx -c config.json
  
  # Resubmit with selective steps (by index)
  %(prog)s resubmit wf-xxxxx -c config.json -u "0-5,10,15-17"
  
  # List available steps for download
  %(prog)s download wf-xxxxx -l
  
  # Download all artifacts from all steps
  %(prog)s download wf-xxxxx -o ./my-downloads
  
  # Download specific steps by index
  %(prog)s download wf-xxxxx -s "0-5,10" -o ./my-downloads
  
  # Download only outputs (no inputs)
  %(prog)s download wf-xxxxx --no-inputs -s "0-5"
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Submit command
    submit_parser = subparsers.add_parser('submit', help='Submit a new workflow')
    submit_parser.add_argument(
        '-c', '--config',
        type=Path,
        required=True,
        help='Path to JSON configuration file'
    )
    submit_parser.add_argument(
        '--save-wf-id',
        type=Path,
        help='Save workflow ID to file'
    )
    submit_parser.set_defaults(func=submit_workflow)
    
    
    # Resubmit command
    resubmit_parser = subparsers.add_parser('resubmit', help='Resubmit workflow with selective step reuse')
    resubmit_parser.add_argument(
        'workflow_id',
        type=str,
        help='Workflow ID to reuse steps from'
    )
    resubmit_parser.add_argument(
        '-c', '--config',
        type=Path,
        required=True,
        help='Path to JSON configuration file'
    )
    resubmit_parser.add_argument(
        '-u', '--indices',
        type=str,
        help='Step indices to reuse. Supports comma-separated values and ranges (e.g., "0-5,10,15-17" or "0-5, 10, 15-17"). If not specified, all steps are reused.'
    )
    resubmit_parser.add_argument(
        '-l', '--list-steps',
        action='store_true',
        help='List available steps and exit without submitting'
    )
    resubmit_parser.add_argument(
        '-f','--fold',
        action='store_true',
        help='Fold step keys by super operation when reusing steps'
    )
    resubmit_parser.add_argument(
        '--reuse-all-steps',
        action='store_true',
        help='Include unsuccessful steps when listing/reusing'
    )
    resubmit_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    resubmit_parser.add_argument(
        '--save-wf-id',
        type=Path,
        help='Save new workflow ID to file'
    )
    
    # Parse indices if provided
    def parse_indices(args):
        if not args.indices:
            args.step_indices = None
            return
        
        args.step_indices = parse_index_string(args.indices)
    
    resubmit_parser.set_defaults(func=lambda args: (parse_indices(args), resubmit_workflow(args)))
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download artifacts from workflow steps')
    download_parser.add_argument(
        'workflow_id',
        type=str,
        help='Workflow ID to download artifacts from'
    )
    download_parser.add_argument(
        '-c', '--config',
        type=Path,
        required=True,
        help='Path to JSON configuration file'
    )
    download_parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./downloads',
        help='Output directory for downloaded artifacts (default: ./downloads)'
    )
    download_parser.add_argument(
        '-s', '--indices',
        type=str,
        help='Step indices to download. Supports comma-separated values and ranges (e.g., "0-5,10,15-17"). If not specified, all steps are downloaded.'
    )
    download_parser.add_argument(
        '-l', '--list-steps',
        action='store_true',
        help='List available steps and exit without downloading'
    )
    download_parser.add_argument(
        '--inputs',
        action='store_true',
        default=True,
        help='Download input artifacts (default: True)'
    )
    download_parser.add_argument(
        '--outputs',
        action='store_true',
        default=True,
        help='Download output artifacts (default: True)'
    )
    download_parser.add_argument(
        '--no-inputs',
        action='store_false',
        dest='inputs',
        help='Do not download input artifacts'
    )
    download_parser.add_argument(
        '--no-outputs',
        action='store_false',
        dest='outputs',
        help='Do not download output artifacts'
    )
    download_parser.add_argument(
        '--include-failed',
        action='store_true',
        help='Include failed/unsuccessful steps when listing/downloading'
    )
    download_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    
    # Parse indices for download command
    def parse_download_indices(args):
        if not hasattr(args, 'indices') or not args.indices:
            args.step_indices = None
            return
        args.step_indices = parse_index_string(args.indices)
    
    download_parser.set_defaults(func=lambda args: (parse_download_indices(args), download_workflow(args)))
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
