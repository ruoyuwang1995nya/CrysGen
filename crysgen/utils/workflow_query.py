from crysgen.superop.evaluate.solid_electrolyte import SolidElectrolyteMatterGen
import dflow
from dflow import Workflow, Step, upload_artifact
from pathlib import Path
from crysgen.op.fp.vasp_input import VaspInputs
import ase
import crysgen
from typing import List
import re

# find successful steps
def successful_step_keys(wf, unsuccessful_step_keys: bool = False):
    """[From DPGEN2] Get the keys of all successful steps in the workflow.

    Args:
        wf (_type_): The workflow object.
        unsuccessful_step_keys (bool, optional): If True, include keys of unsuccessful steps. Defaults to False.

    Returns:
        list: A list of successful step keys.
    """
    all_step_keys = []
    steps = wf.query_step()
    # For reused steps whose startedAt are identical, sort them by key
    steps.sort(key=lambda x: "%s-%s" % (x.startedAt, x.key))
    for step in steps:
        if not unsuccessful_step_keys:
            if step.key is not None and step.phase == "Succeeded":
                all_step_keys.append(step.key)
        else:
            if step.key is not None:
                all_step_keys.append(step.key)
    return all_step_keys

def find_slice_ranges(
    keys: List[str],
    sliced_subkey: str,
):
    """
    find range of sliced OPs that matches the pattern 'iter-[0-9]*--{sliced_subkey}-[0-9]*'
    """
    found_range = []
    tmp_range = []
    status = "not-found"
    for idx, ii in enumerate(keys):
        if status == "not-found":
            if re.match(f".*--{sliced_subkey}-[0-9]*", ii):
                status = "found"
                tmp_range.append(idx)
        elif status == "found":
            if not (
                re.match(f".*--{sliced_subkey}-[0-9]*", ii)
            ):
                status = "not-found"
                tmp_range.append(idx)
                found_range.append(tmp_range)
                tmp_range = []
        else:
            raise RuntimeError(f"unknown status {status}, terrible error")
    return found_range

def _sort_slice_ops(keys, sliced_subkey):
    found_range = find_slice_ranges(keys, sliced_subkey)
    for ii in found_range:
        keys[ii[0] : ii[1]] = sorted(keys[ii[0] : ii[1]])
    return keys

def sort_slice_ops(
    keys: List[str],
    sliced_subkey: List[str],
):
    """
    sort the keys of the sliced ops. the keys of the sliced ops contains sliced_subkey
    """
    if isinstance(sliced_subkey, str):
        sliced_subkey = [sliced_subkey]
    for ii in sliced_subkey:
        keys = _sort_slice_ops(keys, ii)
    return keys

def get_resubmit_keys(wf, unsuccessful_step_keys: bool = False):
    """[From DPGEN2] Get the keys of all steps in the workflow for resubmission.
    """
    all_step_keys = successful_step_keys(wf, unsuccessful_step_keys)

    all_step_keys = sort_slice_ops(
        all_step_keys,
        ["run-vasp","ion-md"],
    )
    return all_step_keys
