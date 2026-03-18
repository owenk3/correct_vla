"""
config.py

Loads correctvla/config.yaml and exposes config sections.
"""
import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")


def load_config() -> dict:
    with open(os.path.abspath(_CONFIG_PATH)) as f:
        return yaml.safe_load(f)


def get_vlm_config() -> dict:
    return load_config()["vlm"]


def get_pi05_config() -> dict:
    return load_config()["pi05"]


def get_eval_config() -> dict:
    return load_config()["eval"]
