import pathlib
from typing import Dict
import yaml

CONFIG_PATH = pathlib.Path(__file__).parent / "config.yaml"


def load_config() -> Dict:
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    return config
