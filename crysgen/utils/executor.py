

def _pop_executor(config: Dict) -> (Dict, Dict, Any):
    """Pop out executor related configs from step config."""
    config = deepcopy(config)
    template_config = config.pop("template_config", {})
    executor_config = config.pop("executor_config", {})
    executor = init_executor(executor_config)
    return config, template_config, executor