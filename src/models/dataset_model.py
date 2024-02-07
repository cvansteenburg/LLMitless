from enum import StrEnum
from typing import Callable

from fastapi import HTTPException, status

from src.parsers.html_parse import PARSE_FNS
from src.utils.file_ops import DatasetFileFormatNames


class InputDocFormat(StrEnum):
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


async def sum_parser_selector(input_doc_format: InputDocFormat) -> Callable[[str], str]:
    match input_doc_format:
        case InputDocFormat.HTML:
            return PARSE_FNS["markdownify_html_to_md"]
        case InputDocFormat.MARKDOWN:
            return PARSE_FNS["passthrough"]
        case InputDocFormat.TEXT:
            return PARSE_FNS["passthrough"]
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid input_doc_format",
            )


def input_format_from_dataset_file_format(
    dataset_file_format: DatasetFileFormatNames,
) -> InputDocFormat:
    match dataset_file_format:
        case DatasetFileFormatNames.HTML:
            return InputDocFormat.HTML
        case DatasetFileFormatNames.RAW:
            return InputDocFormat.TEXT
        case DatasetFileFormatNames.TXT:
            return InputDocFormat.TEXT
        case _:
            raise ValueError(f"Unrecognized dataset_file_format: {dataset_file_format}")
