from typing import Dict

def dict_to_mattergen_args(config: Dict) -> list:
    """
    Convert a nested dictionary into a list of CLI arguments in the form of key=value.
    """
    def flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key))
            else:
                items.append((new_key, v))
        return items

    # Flatten the dictionary and format as key=value
    return [f"{key}={value}" for key, value in flatten_dict(config)]