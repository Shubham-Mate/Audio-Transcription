import pathlib
from typing import Dict
import yaml
from ..utils.paths import CONFIG_PATH


def load_config() -> Dict:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config
