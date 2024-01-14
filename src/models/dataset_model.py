from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import datasets


class DatasetFileFormatNames(StrEnum):
    HTML = "html-message.html"
    JSON = "json-body.json"
    RAW = "raw-source.txt"
    TXT = "text-message.txt"

# Get dataset directory
root_directory = Path(datasets.__path__[0]).resolve()

def build_mapping(directory: Path) -> dict:
    mapping: dict = {}
    for item in directory.iterdir():
        # Skip hidden and private
        if item.name.startswith(('.', '__')):
            continue
        if item.is_dir():
            mapping[item.name] = build_mapping(item)
        else:
            mapping[item.stem] = item
    return mapping

_dataset_map = build_mapping(root_directory)

DATASET_MAP = MappingProxyType(_dataset_map)