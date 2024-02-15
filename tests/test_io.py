# from datetime import datetime
# from pathlib import Path
# from unittest.mock import patch

import os

import pytest

# import time_machine
from fastapi import HTTPException

from src.models.dataset_model import DatasetFileFormatNames
from src.utils.file_operations import (
    FileFilter,
    filter_files,
)
from tests import dataset_for_testing

print(f"DATASET FOR TESTING: {dataset_for_testing.__path__}")

print(f"TESTING ENV: {os.environ}")

# test filter_files with tests/dataset_for_testing

# @pytest.fixture
# def dataset_path():
#     return Path(dataset_for_testing.__path__[0])


@pytest.fixture(
    params=[
        (DatasetFileFormatNames.HTML, "html-message.html"),
        (DatasetFileFormatNames.TXT, "text-message.txt"),
        (DatasetFileFormatNames.RAW, "raw-source.txt"),
    ]
)
def file_format_checks(request):
    return request.param


@pytest.fixture
def file_filter():
    return FileFilter(
        collection_digits="000",
        title_digits=["001", "002"],
        file_format=DatasetFileFormatNames.HTML,
    )


def test_no_matching_collection_digits():
    filter_inputs = FileFilter(collection_digits="999")

    with pytest.raises(HTTPException) as exc_info:
        filter_files(filter_inputs, test_root=True)
        assert exc_info.value.status_code == 404
        assert "Could not find collection 999" in exc_info.value.detail


def test_no_title_digits_provided_returns_all_files(file_format_checks):
    file_format, expected_file_name = file_format_checks
    expected_count = 4  # Expecting 4 HTML files, one from each test collection

    filter_inputs = FileFilter(
        collection_digits="000", title_digits=[], file_format=file_format
    )
    result = filter_files(filter_inputs, test_root=True)

    assert (
        len(result) == expected_count
    ), f"Expected {expected_count} files, got {len(result)}."

    for file_path in result:
        assert file_path.exists(), f"File {file_path} does not exist."
        assert (
            file_path.name == expected_file_name
        ), f"File {file_path} does not match the expected file format."
        assert file_path.is_file(), f"Path {file_path} is not a file."


# Test when both collection_digits and title_digits are provided and match subdirectories
def test_matching_collection_and_multiple_title_digits(file_format_checks):
    file_format, expected_file_name = file_format_checks
    expected_count = 2
    filter_inputs = FileFilter(
        collection_digits="000", title_digits=["001", "999"], file_format=file_format
    )
    result = filter_files(filter_inputs, test_root=True)

    assert (
        len(result) == expected_count
    ), f"Expected {expected_count} files, got {len(result)}."

    for file_path in result:
        assert file_path.exists(), f"File {file_path} does not exist."
        assert (
            expected_file_name in file_path.name
        ), f"File {file_path} does not match the expected file name."
        assert file_path.is_file(), f"Path {file_path} is not a file."


# Test when the file_format specified matches files in the directories
def test_existing_file_format(file_format_checks):
    file_format, expected_file_name = file_format_checks
    filter_inputs = FileFilter(collection_digits="000", file_format=file_format)
    result = filter_files(filter_inputs, test_root=True)

    assert any(
        file_path.exists() for file_path in result
    ), "Expected at least one file to exist."
    assert any(
        expected_file_name in file_path.name for file_path in result
    ), f"Expected files to match the {file_format.value} format."


# Should raise an error when no matches are found
def test_no_files_matching_file_format():
    filter_inputs = FileFilter(
        collection_digits="001",
        title_digits=["001"],
        file_format=DatasetFileFormatNames.HTML,
    )

    with pytest.raises(HTTPException) as exc_info:
        filter_files(filter_inputs, test_root=True)
        assert exc_info.value.status_code == 404
        assert "Could not find any files matching" in exc_info.value.detail


# Test when title_digits is provided but does not match any subdirectory
def test_no_files_matching_title_digits(file_format_checks):
    file_format, _ = file_format_checks
    filter_inputs = FileFilter(
        collection_digits="001", title_digits=["999"], file_format=file_format
    )

    with pytest.raises(HTTPException) as exc_info:
        filter_files(filter_inputs, test_root=True)
        assert exc_info.value.status_code == 404
        assert "Could not find any title directories matching" in exc_info.value.detail
