from datetime import datetime
from enum import StrEnum
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Protocol, overload

import tiktoken
from dotenv import load_dotenv
from fastapi import HTTPException, status
from langchain_core.documents import Document
from pydantic import BaseModel, Field, field_validator

import datasets
from src.models.chain_configs import ModelList
from src.models.dataset_model import DatasetFileFormatNames
from src.services import output

load_dotenv()

logger = getLogger(__name__)

class SummarizationTestPrompt(StrEnum):
    PASSTHROUGH = (
        "Repeat the following input verbatim, without any extra words and without any"
        " conversational words meant for me:"
    )

# NOTE: Current implementation doesn't count tokens in metadata, which may be added to LLM context later
def count_tokens(
    source: Document | str | list[Document | str],
    model: str = ModelList.GPT_3_5_TURBO.value,
) -> int:
    encoding = tiktoken.encoding_for_model(model)
    if isinstance(source, list):
        return sum(
            (
                len(encoding.encode(item.page_content))
                if isinstance(item, Document)
                else len(encoding.encode(item))
            )
            for item in source
        )
    elif isinstance(source, Document):
        return len(encoding.encode(source.page_content))
    elif isinstance(source, str):
        return len(encoding.encode(source))
    else:
        raise ValueError(
            f"Must be Document, str, or list of those. Got type: {type(source)}"
        )


class DocumentContents(BaseModel):
    """Class for storing a piece of text and associated metadata."""

    page_content: str = Field(
        ..., title="Page content", description="Content to summarize"
    )
    metadata: dict | None = Field(
        title="Metadata",
        description=(
            "Arbitrary metadata about the page content (e.g., source, relationships to"
            " other documents, etc.)."
        ),
    )


def sources_to_docs(sources: list[DocumentContents]) -> list[Document]:
    return [
        Document(
            page_content=source.page_content,
            metadata=source.metadata if source.metadata else dict(),
        )
        for source in sources
    ]


# from langchain.chains.combine_documents.reduce.py
# class import was memory intensive so we use directly
class CombineDocsProtocol(Protocol):
    """Interface for the combine_docs method."""

    def __call__(self, docs: list[Document], **kwargs: Any) -> str:
        """Interface for the combine_docs method."""

# from langchain.chains.combine_documents.reduce.py
# class import was memory intensive so we use directly
def collapse_docs(
    docs: list[Document],
    combine_document_func: CombineDocsProtocol,
    **kwargs: Any,
) -> Document:
    """Execute a collapse function on a set of documents and merge their metadatas.

    Args:
        docs: A list of Documents to combine.
        combine_document_func: A function that takes in a list of Documents and
            optionally addition keyword parameters and combines them into a single
            string.
        **kwargs: Arbitrary additional keyword params to pass to the
            combine_document_func.

    Returns:
        A single Document with the output of combine_document_func for the page content
            and the combined metadata's of all the input documents. All metadata values
            are strings, and where there are overlapping keys across documents the
            values are joined by ", ".
    """
    result = combine_document_func(docs, **kwargs)
    combined_metadata = {k: str(v) for k, v in docs[0].metadata.items()}
    for doc in docs[1:]:
        for k, v in doc.metadata.items():
            if k in combined_metadata:
                combined_metadata[k] += f", {v}"
            else:
                combined_metadata[k] = str(v)
    return Document(page_content=result, metadata=combined_metadata)


# from langchain.chains.combine_documents.reduce.py
# class import was memory intensive so we use directly
def split_list_of_docs(
    docs: list[Document], length_func: Callable, token_max: int, **kwargs: Any
) -> list[list[Document]]:
    """Split Documents into subsets that each meet a cumulative length constraint.

    Args:
        docs: The full list of Documents.
        length_func: Function for computing the cumulative length of a set of Documents.
        token_max: The maximum cumulative length of any subset of Documents.
        **kwargs: Arbitrary additional keyword params to pass to each call of the
            length_func.

    Returns:
        A List[List[Document]].
    """
    new_result_doc_list = []
    _sub_result_docs = []
    for doc in docs:
        _sub_result_docs.append(doc)
        _num_tokens = length_func(_sub_result_docs, **kwargs)
        if _num_tokens > token_max:
            if len(_sub_result_docs) == 1:
                raise ValueError(
                    "A single document was longer than the context length,"
                    " we cannot handle this."
                )
            new_result_doc_list.append(_sub_result_docs[:-1])
            _sub_result_docs = _sub_result_docs[-1:]
    new_result_doc_list.append(_sub_result_docs)
    return new_result_doc_list


# chunk and load
async def split_large_docs(
    docs: list[Document],
    len_finder_fn: Callable[..., int],
    max_doc_size: int, # in tokens
    split_on_value: str = "\n\n",
    chars_per_token_est: float = 3.5,
) -> list[Document]:
    # if doc is larger than max_chunk_size, split on nearest separator that yields max_chunk_size, maintaining metadata"
    max_doc_size_chars = (max_doc_size * chars_per_token_est) // 1
    docs_list = []
    for doc in docs:
        if len(doc.page_content) > max_doc_size_chars:
            _doc_chunks = doc.page_content.split(sep=split_on_value)
            _doc_under_construction: list[str] = []
            _doc_under_construction_size = 0
            _metadata = doc.metadata.copy()
            _page = 1
            _finalized_docs: list[Document] = []

            # split long doc on newlines, and construct several docs of max_size from those chunks
            for chunk in _doc_chunks:
                _chunk_size = len(chunk)

                # TODO: Add addl chunker for strings without enough split_on_value separators to allow them to meet min size requirement
                if _chunk_size > max_doc_size_chars:
                    raise ValueError(
                        f"Minimum chunk size {_chunk_size} is larger than max doc size"
                        f" {max_doc_size}. We split docs that are too long, but this"
                        " failed. Maybe the separator doesn't exist in the doc? Try"
                        " changing the split_on_value."
                    )

                if _doc_under_construction_size + _chunk_size >= max_doc_size_chars:
                    _metadata["page"] = _page
                    _finalized_docs.append(
                        Document(
                            page_content="".join(_doc_under_construction),
                            metadata=_metadata.copy(),
                        )
                    )
                    _doc_under_construction = []
                    _doc_under_construction_size = 0
                    _page += 1

                _doc_under_construction.append(chunk)  # add chunk
                _doc_under_construction_size += _chunk_size

            # construct doc from remaining chunks
            if _doc_under_construction:
                _metadata["page"] = _page
                _finalized_docs.append(
                    Document(
                        page_content="".join(_doc_under_construction),
                        metadata=_metadata.copy(),
                    )
                )

            _page = 0

            # add finalized docs to list of docs
            docs_list.extend(_finalized_docs)

        else:
            docs_list.append(doc)

    token_normalized_docs_list = await token_split_docs(docs_list, len_finder_fn, max_doc_size, split_on_value)

    return token_normalized_docs_list


# chunk based on text, check via tiktoken - this minimizes use of (slower) tiktoken
async def token_split_docs(
    docs: list[Document],
    len_finder_fn: Callable[..., int],
    max_doc_size: int,
    split_on_value: str = "\n\n",
) -> list[Document]:
    # if doc is larger than max_chunk_size, split on nearest separator that yields max_chunk_size, maintaining metadata"
    docs_list = []
    for doc in docs:
        if len_finder_fn(doc.page_content) > max_doc_size:
            _doc_chunks = doc.page_content.split(sep=split_on_value)
            _doc_under_construction: list[str] = []
            _doc_under_construction_size = 0
            _metadata = doc.metadata.copy()
            _page = 1
            _finalized_docs: list[Document] = []

            # split long doc on newlines, and construct several docs of max_size from those chunks
            for chunk in _doc_chunks:
                _chunk_size = len_finder_fn(chunk)

                # TODO: Add addl chunker for strings without enough split_on_value separators to allow them to meet min size requirement
                if _chunk_size > max_doc_size:
                    raise ValueError(
                        f"Minimum chunk size {_chunk_size} is larger than max doc size"
                        f" {max_doc_size}. We split docs that are too long, but this"
                        " failed. Maybe the separator doesn't exist in the doc? Try"
                        " changing the split_on_value."
                    )

                if _doc_under_construction_size + _chunk_size >= max_doc_size:
                    _metadata["page"] = _page
                    _finalized_docs.append(
                        Document(
                            page_content="".join(_doc_under_construction),
                            metadata=_metadata.copy(),
                        )
                    )
                    _doc_under_construction = []
                    _doc_under_construction_size = 0
                    _page += 1

                _doc_under_construction.append(chunk)  # add chunk
                _doc_under_construction_size += _chunk_size

            # construct doc from remaining chunks
            if _doc_under_construction:
                _metadata["page"] = _page
                _finalized_docs.append(
                    Document(
                        page_content="".join(_doc_under_construction),
                        metadata=_metadata.copy(),
                    )
                )

            _page = 0

            # add finalized docs to list of docs
            docs_list.extend(_finalized_docs)

        else:
            docs_list.append(doc)

    return docs_list


class FileFilter(BaseModel):
    collection_digits: str = Field(
        ...,
        title="Collection digits",
        description='Usually a 3 digit number expressed as a string eg. "010"',
    )
    title_digits: list[str] | None = Field(
        default= None,
        title="Title digits",
        description=(
            'A list of usually 3 digit numbers expressed as a strings eg. ["001",'
            ' "002", "009"]'
        ),
    )
    file_format: DatasetFileFormatNames = DatasetFileFormatNames.HTML

    @field_validator ("collection_digits")
    @classmethod
    def validate_collection_digits(cls, v: str):
        if v and not v.isdigit() or not 0 <= int(v) <= 99999:
            raise ValueError("collection_digits must be a string representing an integer between 0 and 99999")
        return v

    @field_validator("title_digits")
    @classmethod
    def validate_title_digits(cls, v: list[str] | None):
        v = None if v == [] else v
        if v is not None:
            for i in v:
                if not i.isdigit() or not 0 <= int(i) <= 99999:
                    raise ValueError("Each int in title_digits must be a string representing an integer between 0 and 99999")
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
        output_path = Path(output.__path__[0]).resolve()

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


def parse_files(
    input_files: list[DocumentContents],
    parse_function: Callable[[str], str] = (lambda x: x),
    **kwargs: Any,
) -> list[DocumentContents]:
    """
    Parses files using the provided parse_function.
    output_format should match output format of chosen parse_function
    """
    docs = []
    _file_count = 0
    for file in input_files:
        _file_count += 1
        _metadata = (
            file.metadata if file.metadata else {"title": f"Doc num {_file_count}"}
        )
        _parsed_content = parse_function(file.page_content, **kwargs)
        docs.append(DocumentContents(page_content=_parsed_content, metadata=_metadata))

    return docs


def combine_document_content(
    doc_list: list[Document], metadata_to_include: list[str] | None = None
) -> str:
    """Combine the content of the Documents in the doc_list into one string.
    metadata_to_include is a comma-separated string of metadata keys: the corresponding
    values for those keys will be included at the beginning of each combined content
    segment, and will be readable by the LLM.

    Example:

    metadata_to_include = ["source", "part"]

    OUTPUT:
    --Source 5 Part 2--
    Text content

    """
    content_components = []
    doc_count = 1
    total_docs = len(doc_list)
    header_marker = "--"
    header_to_content_transition = "\n\n"
    post_content_transition = "\n\n"

    for doc in doc_list:
        if metadata_to_include is None:
            header_content = f"Source doc {doc_count} of {total_docs}"
        else:
            header_content = ", ".join(
                f"{key}: {doc.metadata[key]}" for key in metadata_to_include
            )

        doc_text = doc.page_content
        content_components.append(
            f"{header_marker} {header_content} {header_marker}{header_to_content_transition}{doc_text}{post_content_transition}"
        )

    content_as_str = "".join(content_components)

    return content_as_str


def consolidate_lists(
    source_lists: list[list[Document]], combine_doc_fn, **kwargs
) -> list[Document]:
    collapsed_docs = []
    for list in source_lists:
        collapsed_docs.append(collapse_docs(list, combine_doc_fn, **kwargs))
    return collapsed_docs


@overload
async def transform_raw_docs(
    input_files: list[Path],
    parse_fn: Callable[[str], str],
    max_tokens_per_doc: int,
    metadata_to_include: list[str] | None = None,
    **kwargs,
) -> list[Document]: ...
@overload
async def transform_raw_docs(
    input_files: list[DocumentContents],
    parse_fn: Callable[[str], str],
    max_tokens_per_doc: int,
    metadata_to_include: list[str] | None = None,
    **kwargs,
) -> list[Document]: ...
async def transform_raw_docs(
    input_files,
    parse_fn: Callable[[str], str],
    max_tokens_per_doc: int,
    metadata_to_include: list[str] | None = None,
    **kwargs,
) -> list[Document]:
    try:
        if isinstance(input_files[0], Path):
            parsed_input_files = parse_files_from_paths(input_files, parse_fn, **kwargs)
        else:
            parsed_input_files = parse_files(input_files, parse_fn, **kwargs)

    except TypeError as e:
        logger.error(
            f"Error parsing files in transform_raw_docs. TypeError {e}", exc_info=e
        )
        raise TypeError("Expected a list of a single type as input")

    docs = sources_to_docs(parsed_input_files)
    sized_docs = await split_large_docs(docs, count_tokens, max_tokens_per_doc)

    consolidated_lists = split_list_of_docs(
        sized_docs, count_tokens, max_tokens_per_doc, **kwargs
    )

    consolidated_docs = consolidate_lists(
        consolidated_lists,
        combine_document_content,
        metadata_to_include=metadata_to_include,
        **kwargs,
    )

    return consolidated_docs