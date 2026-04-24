import os
import yaml
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
from src.saferesponse_engine import logger


class ConfigBox:
    """
    Minimal ConfigBox implementation (no external dependency).
    Provides dot-notation access to dict keys, recursively.
    """

    def __init__(self, data: Dict):
        if not isinstance(data, dict):
            raise TypeError(f"ConfigBox expects a dict, got {type(data)}")

        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigBox(value)
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        # Optional: still allow dict-style access if you want
        return getattr(self, key)

    def to_dict(self) -> Dict:
        # Optional: convert back to plain dict
        out = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigBox):
                out[key] = value.to_dict()
            else:
                out[key] = value
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns a ConfigBox.

    Args:
        path_to_yaml (Path): Path to YAML file

    Raises:
        ValueError: If YAML file is empty
        FileNotFoundError: If YAML path does not exist

    Returns:
        ConfigBox: Parsed YAML content with dot access
    """
    if not path_to_yaml.exists():
        raise FileNotFoundError(f"YAML file not found: {path_to_yaml}")

    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        if content is None:
            raise ValueError(f"YAML file is empty: {path_to_yaml}")

        logger.info(f"YAML file loaded successfully: {path_to_yaml}")
        return ConfigBox(content)

    except Exception as e:
        logger.exception(e)
        raise e


def create_directories(paths: List[Path], verbose: bool = True):
    """
    Create directories if they do not exist.

    Args:
        paths (List[Path]): List of directory paths
        verbose (bool): Whether to log directory creation
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


def save_json(path: Path, data: Dict):
    """
    Save dict data to a JSON file.

    Args:
        path (Path): Path to JSON file
        data (Dict): Data to save
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

def load_json(path: Path) -> ConfigBox:
    """
    Load JSON data and return ConfigBox.

    Args:
        path (Path): Path to JSON file

    Returns:
        ConfigBox: JSON content with dot access
    """
    with open(path, "r") as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


def save_bin(data: Any, path: Path):
    """
    Save object as binary using pickle.

    Args:
        data (Any): Object to save
        path (Path): Output file path (e.g., model.pkl)
    """
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"Binary file (pickle) saved at: {path}")


def load_bin(path: Path) -> Any:
    """
    Load object from binary using pickle.

    Args:
        path (Path): Input file path

    Returns:
        Any: Loaded object
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info(f"Binary file (pickle) loaded from: {path}")
    return data

def get_size(path: Path) -> str:
    """
    Get file size in KB.

    Args:
        path (Path): File path

    Returns:
        str: Approx size in KB string
    """
    size_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_kb} KB"