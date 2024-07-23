"""Utilities package for ClickhouseRAG."""

import importlib
import json
import os
from typing import Any


def check_installed(*libraries: str) -> None:
    """Check if the required libraries are installed.

    Args:
    ----
        libraries (str): Libraries to check.

    Raises:
    ------
        ImportError: If a required library is not installed.
    """
    for lib in libraries:
        try:
            importlib.import_module(lib)
        except ImportError as e:
            raise ImportError(f"Library '{lib}' is not installed. Please install it to use this feature.") from e

def get_format_from_path(path: str) -> str:
    """Get the backup format from the file path.

    Args:
    ----
        path (str): The file path.

    Returns:
    -------
        str: The backup format inferred from the file extension.

    Raises:
    ------
        ValueError: If the file extension does not match a supported format.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext:
        return ext.lstrip(".")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def load_json(path: str) -> Any:
    with open(path, "r") as file:
        return json.load(file)

def save_json(data: Any, path: str) -> None:
    with open(path, "w") as file:
        json.dump(data, file)
