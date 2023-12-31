from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import time_machine

from src.models.dataset_model import DatasetFileFormatNames
from src.services.io import (
    filter_files,
    parse_files,
    read_file_content,
)
from tests import dataset_for_testing

print(f'DATASET FOR TESTING: {dataset_for_testing.__path__}')

# test filter_files with tests/dataset_for_testing

@pytest.fixture
def dataset_path():
    return Path(dataset_for_testing.__path__[0])

@pytest.fixture(params=[
    (DatasetFileFormatNames.HTML, 'html-message.html'),
    (DatasetFileFormatNames.TXT, 'text-message.txt'),
    (DatasetFileFormatNames.JSON, 'json-body.json'),
    (DatasetFileFormatNames.RAW, 'raw-source.txt')
])
def file_format_checks(request):
    return request.param

def test_no_matching_collection_digits(dataset_path):
    result = filter_files('999', dataset_path)
    assert len(result) == 0, "Expected no files to be returned when no collection digits match."

def test_no_title_digits_provided_returns_all_files(dataset_path, file_format_checks):
    file_format, expected_file_name = file_format_checks
    expected_count = 4  # Expecting 4 HTML files, one from each test collection
    result = filter_files('000', dataset_path, file_format=file_format)

    assert len(result) == expected_count, f"Expected {expected_count} files, got {len(result)}."

    for file_path in result:
        assert file_path.exists(), f"File {file_path} does not exist."
        assert file_path.name == expected_file_name, f"File {file_path} does not match the expected file format."
        assert file_path.is_file(), f"Path {file_path} is not a file."

# Test when both collection_digits and title_digits are provided and match subdirectories
def test_matching_collection_and_multiple_title_digits(dataset_path, file_format_checks):
    file_format, expected_file_name = file_format_checks
    expected_count = 2
    result = filter_files('000', dataset_path, title_digits=['001', '999'], file_format=file_format)
    
    assert len(result) == expected_count, f"Expected {expected_count} files, got {len(result)}."
    
    for file_path in result:
        assert file_path.exists(), f"File {file_path} does not exist."
        assert expected_file_name in file_path.name, f"File {file_path} does not match the expected file name."
        assert file_path.is_file(), f"Path {file_path} is not a file."

# Test when the file_format specified matches files in the directories
def test_existing_file_format(dataset_path, file_format_checks):
    file_format, expected_file_name = file_format_checks
    result = filter_files('000', dataset_path, file_format=file_format)
    
    assert any(file_path.exists() for file_path in result), "Expected at least one file to exist."
    assert any(expected_file_name in file_path.name for file_path in result), \
        f"Expected files to match the {file_format.value} format."

# Test ensuring that it returns an empty list when no matches are found
def test_no_matches_found(dataset_path):
    result = filter_files('999', dataset_path, title_digits=['nonexistent_title'], file_format=DatasetFileFormatNames.HTML)
    
    assert len(result) == 0, "Expected no files to be returned when no matches are found."

# Test when title_digits is provided but does not match any subdirectory
def test_no_files_matching_title_digits(dataset_path, file_format_checks):
    file_format, _ = file_format_checks
    result = filter_files('001', dataset_path, title_digits=['999'], file_format=file_format)
    
    assert len(result) == 0, "Expected no files to be returned when title digits do not match any subdirectory."



@pytest.fixture
def input_files(tmp_path) -> list[Path]:
    # Generate some dummy files to act as input
    file_paths = []
    for i in range(3):
        file_path = tmp_path / f"input_{i}.txt"
        file_path.write_text(f"Content of file {i}")
        file_paths.append(file_path)
    return file_paths

def test_parse_files(tmp_path, input_files):
    read_file_content_fn = f"src.services.io.{read_file_content.__name__}"

    with patch(read_file_content_fn) as mock_read, \
        time_machine.travel(0, tick=False):
        
        output_file_format = 'html'
        output_base_name = 'test_output'
        
        mock_read.side_effect = lambda x: f"READ {x}"
        
        parse_files(
            input_file_paths=input_files,
            parse_function=lambda content, **kwargs: f"Parsed {content}",
            output_path=tmp_path,
            output_base_name=output_base_name,
            output_format=output_file_format
        )

        expected_output_file_name = "test_output".join(
            datetime.now().isoformat(timespec="milliseconds").split("T")
        )

        file_pattern = f"{expected_output_file_name}.{output_file_format}"

        matching_files = list(tmp_path.glob(file_pattern))

        assert len(matching_files) == 1, f"No files matching pattern {file_pattern} found in {tmp_path}"

        output_file_path = matching_files[0]
        with open(output_file_path, "r") as f:
            output_content = f.read()

        for file_path in input_files:
            title_name = file_path.name
            expected_content = f"Title: {title_name}\nParsed READ {file_path}\n\n"
            assert expected_content in output_content, f"Expected content for {title_name} not found in output."

        for file_path in input_files:
            mock_read.assert_any_call(file_path)
