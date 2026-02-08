import os
import dflow
from crysgen.utils.dflow_config import (
    workflow_config_from_dict,
)
from crysgen.utils.bohrium import (
    bohrium_config_from_dict,
)             
                          
                          
def global_config_workflow(
    wf_config,
):
    # dflow_config, dflow_s3_config
    workflow_config_from_dict(wf_config)

    if os.getenv("DFLOW_DEBUG"):
        dflow.config["mode"] = "debug"
        return None

    # bohrium configuration
    if wf_config.get("bohrium_config") is not None:
        bohrium_config_from_dict(wf_config["bohrium_config"])