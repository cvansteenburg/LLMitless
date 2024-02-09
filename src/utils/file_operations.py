from datetime import datetime
from enum import StrEnum
from logging import getLogger
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, status
from pydantic import BaseModel, Field, field_validator

import datasets
import output_collection
from src.services.doc_operations import DocumentContents

logger = getLogger(f"llmitless.{__name__}")


class DatasetFileFormatNames(StrEnum):
    HTML = "html-message.html"
    RAW = "raw-source.txt"
    TXT = "text-message.txt"


class FileFilter(BaseModel):
    collection_digits: str = Field(
        ...,
        title="Collection digits",
        description='Usually a 3 digit number expressed as a string eg. "010"',
    )
    title_digits: list[str] | None = Field(
        default=None,
        title="Title digits",
        description=(
            'A list of usually 3 digit numbers expressed as a strings eg. ["001",'
            ' "002", "009"]'
        ),
    )
    file_format: DatasetFileFormatNames = DatasetFileFormatNames.HTML

    @field_validator("collection_digits")
    @classmethod
    def validate_collection_digits(cls, v: str):
        if v and not v.isdigit() or not 0 <= int(v) <= 99999:
            raise ValueError(
                "collection_digits must be a string representing an integer between 0 and 99999"
            )
        return v

    @field_validator("title_digits")
    @classmethod
    def validate_title_digits(cls, v: list[str] | None):
        v = None if v == [] else v
        if v is not None:
            for i in v:
                if not i.isdigit() or not 0 <= int(i) <= 99999:
                    raise ValueError(
                        "Each int in title_digits must be a string representing an integer between 0 and 99999"
                    )
        return v


def filter_files(filter_inputs: FileFilter, test_root: bool = False) -> list[Path]:
    """
    Filters and returns a list of file paths from a dataset directory based on the provided criteria.

    The function searches within a dataset directory for subdirectories with leading characters in the name
    that match the `collection_digits`. If `title_digits` is provided, it further narrows down the search to
    include only those subdirectories that match the `title_digits`. The function then collects
    files matching the specified `file_format`.

    Args:
        collection_digits: A string of (usually 3) digits that the dataset collection name should start with.
        title_digits: An optional list of strings containing digits (usually 3) that the title directories
                      within the dataset should start with. If None, all titles in the collection
                      are included.
        file_format: A DatasetFileFormatNames enum member representing the file format to filter.
                     Defaults to DatasetFileFormatNames.HTML.

    Returns:
        A list of pathlib.Path objects representing the filtered file paths.

    Raises:
        None
    """
    _col_digits = filter_inputs.collection_digits
    _title_digits = filter_inputs.title_digits
    _file_format = filter_inputs.file_format.value

    try:
        if test_root:
            # TODO: make this an env var
            from tests import dataset_for_testing as test_data

            # path to tests/dataset_for_testing
            dataset_root = Path(test_data.__path__[0]).resolve()
        else:
            dataset_root = Path(datasets.__path__[0]).resolve()

    except FileNotFoundError:
        logger.error("Dataset path not found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset path not found",
        )

    primary_dir = next(dataset_root.glob(f"{_col_digits}*"), None)

    if primary_dir is None:
        logger.error(
            f"Could not find collection {_col_digits} in dataset {dataset_root}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find collection {_col_digits} in dataset {dataset_root}",
        )

    if _title_digits is None:
        return list(primary_dir.rglob(_file_format))

    filtered_files: list[Path] = []

    for digits in _title_digits:
        target_dirs = [dir for dir in primary_dir.glob(f"{digits}*") if dir.is_dir()]

        if not target_dirs:
            logger.error(
                f"Could not find any title directories matching {digits} in collection {_col_digits} in dataset {dataset_root}"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not find any title directories matching {digits} in collection {_col_digits} in dataset {dataset_root}",
            )

        for dir in target_dirs:
            filtered_files.extend(dir.glob(_file_format))

    if not filtered_files:
        logger.error(
            f"Could not find any files matching {_file_format} in collection {_col_digits} in dataset {dataset_root}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not find any files matching {_file_format} in collection {_col_digits} in dataset {dataset_root}",
        )

    return filtered_files


def read_file_content(file_path: Path) -> str:
    with open(file_path, "r") as content_file:
        return content_file.read()


def parse_files_from_paths(
    input_file_paths: list[Path],
    parse_function: Callable[[str], str] = lambda x: x,
    *,
    write_to_file: bool = False,
    output_path: Path | None = None,
    output_base_name: str = "combined",
    output_format: str = "txt",
    **kwargs: Any,
) -> list[DocumentContents]:
    """
    Reads files from disk, parses them using the provided parse_function, and optionally
    writes them to disk and always returns them.
    """
    if not output_path:
        output_path = Path(output_collection.__path__[0]).resolve()

    output_base_name = output_base_name.join(
        datetime.now().isoformat(timespec="milliseconds").split("T")
    )
    output_file_path = Path(output_path) / f"{output_base_name}.{output_format}"

    docs = []

    if write_to_file:
        with open(output_file_path, "x") as output_file:

            for file_path in input_file_paths:
                _title_name = f"{file_path.parent.name}/{file_path.name}"
                _metadata = {"title": f"{_title_name}"}
                _content = read_file_content(file_path)
                _parsed_content = parse_function(_content, **kwargs)
                output_file.write(f"DOC: {_title_name}\n{_parsed_content}\n\n")
                docs.append(
                    DocumentContents(page_content=_parsed_content, metadata=_metadata)
                )

    else:
        for file_path in input_file_paths:
            _title_name = f"{file_path.parent.name}/{file_path.name}"
            _metadata = {"title": f"{_title_name}"}
            _content = read_file_content(file_path)
            _parsed_content = parse_function(_content, **kwargs)
            docs.append(
                DocumentContents(page_content=_parsed_content, metadata=_metadata)
            )

    return docs
