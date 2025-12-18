#!/usr/bin/env python
"""Entry point for solid electrolyte evaluation workflow."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import ase
import dflow
from dflow import Workflow, Step, upload_artifact

import crysgen
from crysgen.superop.evaluate.solid_electrolyte import SolidElectrolyteMatterGen
from crysgen.op.fp.vasp_input import VaspInputs
from crysgen.utils.workflow_query import get_resubmit_keys


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def setup_python_packages() -> List[str]:
    """Setup Python packages to upload."""
    upload_python_packages = []
    upload_python_packages.extend(list(dflow.__path__))
    upload_python_packages.extend(list(ase.__path__))
    upload_python_packages.extend(list(crysgen.__path__))
    return upload_python_packages

def build_step_configs(config: dict) -> tuple:
    """Build step configurations from config dict."""
    default_step_config = config.get("default_step_config", {
        "executor": None,
        "template_config": {},
    })
    
    vasp_step_config = config.get("fp", default_step_config)
    
    ion_md_step_config = config.get("ion_md", default_step_config)
    
    return default_step_config, vasp_step_config, ion_md_step_config


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
        "structures": upload_artifact(Path(config["structures"])),
        "model": upload_artifact(Path(config["model"])),
    }
    if reference_dataset:= config.get("reference_dataset"):
        artifacts["reference_dataset"] = upload_artifact(Path(reference_dataset))
        
    #artifacts["model"] = upload_artifact(Path(config["model"]))
    return artifacts


def build_workflow(config: dict) -> Workflow:
    """Build workflow from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Workflow object (not yet submitted)
    """
    # Setup Python packages
    upload_python_packages = setup_python_packages()
    
    # Build step configs
    default_step_config, vasp_step_config, ion_md_step_config = build_step_configs(
        config["step_config"])
    
    # Get job configs
    workflow_config = config["jobs"].copy()
    
    # Build VASP config
    vasp_config = workflow_config.pop("vasp", {})
    vasp_config = build_vasp_config(vasp_config)
    
    # Build artifacts
    artifacts_config = config["artifacts"].copy()
    artifacts = build_artifacts(artifacts_config)
    
    # Build SuperOP
    solid_electrolyte_evaluate_op = SolidElectrolyteMatterGen.build(
        name=config.get("superop_name", "solid-electrolyte-evaluate"),
        misc_step_config=default_step_config,
        sun_eval_step_config=default_step_config,
        ff_step_config=ion_md_step_config,
        dft_step_config=vasp_step_config,
        #ion_md_step_config=ion_md_step_config,
        upload_python_packages=upload_python_packages,
    )
    
    # Create workflow
    wf_name = config.get("name", "solid-electrolyte-mattergen")
    wf = Workflow(name=wf_name)
    
    super_steps = Step(
        name=config.get("step_name", "mattergen-eval"),
        template=solid_electrolyte_evaluate_op,
        parameters={
            "name": config.get("task_name", "test"),
            "vasp_config": vasp_config,
            "config": workflow_config,
        },
        artifacts=artifacts
    )
    
    wf.add(super_steps)
    
    return wf


def submit_workflow(args):
    """Submit a new workflow."""
    config = load_config(args.config)
    
    # Setup dflow mode
    dflow.config["mode"] = config.get("dflow_mode", "debug")
    
    # Build workflow
    wf = build_workflow(config)
    
    wf.submit()
    
    print(f"Workflow submitted: {wf.id}")
    print(f"Workflow URL: {wf.url}")
    
    # Save workflow ID
    if args.save_wf_id:
        with open(args.save_wf_id, 'w') as f:
            f.write(wf.id)
        print(f"Workflow ID saved to {args.save_wf_id}")


def query_workflow(args):
    """Query workflow status."""
    wf = Workflow(id=args.workflow_id)
    
    print(f"Workflow ID: {wf.id}")
    print(f"Workflow Name: {wf.name}")
    print(f"Workflow Phase: {wf.phase}")
    print(f"Workflow URL: {wf.url}")
    
    if args.verbose:
        steps = wf.query_step()
        print(f"\nTotal steps: {len(steps)}")
        
        # Group by phase
        phase_counts = {}
        for step in steps:
            phase = step.phase
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        print("\nStep phases:")
        for phase, count in sorted(phase_counts.items()):
            print(f"  {phase}: {count}")
        
        if args.show_steps:
            print("\nAll steps:")
            for step in steps:
                print(f"  {step.key}: {step.phase} (started: {step.startedAt})")


def print_steps(step_keys: List[str]) -> None:
    """Print step names with their index numbers.
    
    Args:
        step_keys: List of step key strings
    """
    for idx, step_key in enumerate(step_keys):
        print(f"  [{idx}] {step_key}")


def parse_index_string(index_str: str) -> List[int]:
    """Parse index string with comma-separated values and ranges.
    
    Supports formats like:
    - "0,1,5" -> [0, 1, 5]
    - "0-5" -> [0, 1, 2, 3, 4, 5]
    - "0-5,10,15-17" -> [0, 1, 2, 3, 4, 5, 10, 15, 16, 17]
    - "0-5, 10, 15-17" (with spaces) -> same as above
    
    Args:
        index_str: String containing indices and/or ranges
        
    Returns:
        List of parsed integer indices
    """
    indices = []
    # Split by comma and strip whitespace
    items = [item.strip() for item in index_str.split(',')]
    
    for item in items:
        if not item:
            continue
        
        if '-' in item:
            # Handle range
            try:
                parts = item.split('-')
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    indices.extend(range(start, end + 1))
                else:
                    print(f"Warning: Invalid range '{item}', skipping")
            except ValueError:
                print(f"Warning: Invalid range '{item}', skipping")
        else:
            # Handle single index
            try:
                indices.append(int(item))
            except ValueError:
                print(f"Warning: Invalid index '{item}', skipping")
    
    return indices


def get_steps_by_indices(step_keys: List[str], indices: Optional[List[int]] = None) -> List[str]:
    """Select steps by index numbers.
    
    Args:
        step_keys: List of all step keys
        indices: List of indices to select, or None for all
    
    Returns:
        List of selected step keys
    """
    if indices is None:
        return step_keys
    
    selected = []
    for idx in indices:
        if 0 <= idx < len(step_keys):
            selected.append(step_keys[idx])
        else:
            print(f"Warning: Index {idx} out of range (0-{len(step_keys)-1}), skipping")
    
    return selected


def list_reusable_steps(args):
    """List steps that can be reused."""
    # Set ARGO server
    config= load_config(args.config)
    dflow.config["mode"] = config.get("dflow_mode", "debug")
    
    wf = Workflow(id=args.workflow_id)
    
    step_keys = get_resubmit_keys(
        wf,
        unsuccessful_step_keys=args.include_unsuccessful
    )
    
    print(f"Reusable steps from workflow {args.workflow_id}:")
    print(f"Total: {len(step_keys)} steps")
    
    print_steps(step_keys)
    
    # Save to file if requested
    if args.save_to:
        with open(args.save_to, 'w') as f:
            json.dump(step_keys, f, indent=2)
        print(f"Step keys saved to {args.save_to}")


def resubmit_workflow(args):
    """Resubmit workflow with selective step reuse."""
    config = load_config(args.config)
    
    # Setup dflow mode
    dflow.config["mode"] = config.get("dflow_mode", "debug")
    
    # Get old workflow
    wf_old = Workflow(id=args.workflow_id)
    
    # Get all reusable steps
    all_step_keys = get_resubmit_keys(
        wf_old,
        unsuccessful_step_keys=args.reuse_all_steps
    )
    print(all_step_keys[0])
    print(f"Found {len(all_step_keys)} reusable steps from workflow {args.workflow_id}")
    
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
    
    if args.verbose:
        print("\nSteps to reuse:")
        print_steps(step_keys)
    
    # Build workflow
    wf = build_workflow(config)
    
    # Submit workflow with step reuse
    reuse_step = wf_old.query_step(key=step_keys)
    print(f"\nSubmitting workflow with {len(step_keys)} reused steps...")
    wf.submit(reuse_step=reuse_step)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Solid Electrolyte Evaluation Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit new workflow
  %(prog)s submit -c config.json
  
  # Query workflow status
  %(prog)s query wf-xxxxx -v
  
  # List reusable steps
  %(prog)s list-steps wf-xxxxx -c config.json
  
  # Resubmit with all steps
  %(prog)s resubmit -c config.json --reuse-workflow wf-xxxxx
  
  # Resubmit with selective steps (by index)
  %(prog)s resubmit -c config.json --reuse-workflow wf-xxxxx -u "0-5,10,15-17"
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
        '--reuse-workflow',
        type=str,
        help='Workflow ID to reuse steps from'
    )
    submit_parser.add_argument(
        '--reuse-all-steps',
        action='store_true',
        help='Reuse all steps including unsuccessful ones'
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
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query workflow status')
    query_parser.add_argument(
        'workflow_id',
        type=str,
        help='Workflow ID to query'
    )
    query_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )
    query_parser.add_argument(
        '--show-steps',
        action='store_true',
        help='Show all step details (requires --verbose)'
    )
    query_parser.set_defaults(func=query_workflow)
    
    # List steps command
    list_parser = subparsers.add_parser('list-steps', help='List reusable steps')
    list_parser.add_argument(
        'workflow_id',
        type=Path,
        help='Workflow ID to list steps from'
    )
    list_parser.add_argument(
        '-c', '--config',
        type=Path,
        required=True,
        help='Path to JSON configuration file'
    )
    list_parser.add_argument(
        '--include-unsuccessful',
        action='store_true',
        help='Include unsuccessful steps'
    )
    list_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print all step keys'
    )
    list_parser.add_argument(
        '--save-to',
        type=Path,
        help='Save step keys to JSON file'
    )
    list_parser.set_defaults(func=list_reusable_steps)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
