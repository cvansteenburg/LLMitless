import os
from enum import StrEnum
from typing import Any, Coroutine

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from langchain.callbacks import get_openai_callback
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.chains.map_reduce import map_reduce
from src.models.dataset_model import DatasetFileFormatNames
from src.parsers.html_parse import PARSE_FNS
from src.services.io import (
    DATASET_PATH,
    DocumentContents,
    SummarizationTestPrompt,
    filter_files,
    transform_raw_docs,
)
from src.utils.logging_init import init_logging

ENV_CONTEXT = os.getenv("ENV_CONTEXT", "development")

# Load the appropriate .env file
env_file = (
    ".env.dev"
    if ENV_CONTEXT == "development"
    else ".env.test" if ENV_CONTEXT == "test" else ".env"
)
load_dotenv(env_file)


app = FastAPI()

CONFIG_FILE = "pyproject.toml"
logger = init_logging(CONFIG_FILE)

app.add_middleware(HTTPSRedirectMiddleware)


# Make sure we're live
@app.get("/")
async def root():
    logger.info("Hello World")
    return {"message": "Hello World"}


class FileFilter(BaseModel):
    collection_digits: str = Field(
        ...,
        title="Collection digits",
        description='Usually a 3 digit number expressed as a string eg. "010"',
    )
    title_digits: list[str] = Field(
        ...,
        title="Title digits",
        description=(
            'A list of usually 3 digit numbers expressed as a strings eg. ["001",'
            ' "002", "009"]'
        ),
    )


class Preprocessor(BaseModel):
    max_tokens_per_doc: int = Field(
        default=3000,
        title="Max tokens per doc",
        description=(
            "The maximum number of tokens to include in each doc that the LLM will"
            " summarize. Change this to according the context window of the LLM and the"
            " length of the prompt."
        ),
    )
    metadata_to_include: list[str] | None = Field(
        default=None,
        title="Metadata passed to LLM",
        description=(
            "In a map-reduce summarization strategy, docs are combined and presented"
            " together to the llm. The metadata keys are included in the combined"
            " documents to give the LLM more context."
        ),
    )


class SummarizeMapReduce(BaseModel):
    core_prompt: str | None = Field(
        default=None,
        title="core prompt",
        description=(
            "Provides a role, context, and instructions for the LLM. LLM will summarize"
            " the content using this prompt. This is a string with brackets around"
            " template variables. At a minimum, prompt should include page_content in"
            " brackets like this: {page_content} where the contents of each summarized"
            " document will be inserted."
        ),
    )
    collapse_prompt: str | None = Field(
        default=None,
        title="collapse prompt",
        description=(
            "In a map-reduce summarization strategy, the LLM will summarize the"
            " summaries to reduce the total amount of text. LLM will use this prompt to"
            " summarize the summaries. Use template variables {core_prompt} to include"
            " the core prompt, and {context} wherever the summaries to summarize will"
            " be inserted."
        ),
    )
    combine_prompt: str | None = Field(
        default=None,
        title="combine prompt",
        description=(
            "In a map-reduce summarization strategy, the last step is for the LLM to"
            " combine the summaries. LLM will use this prompt to combine the summaries."
            " Use template variables {core_prompt} to include the core prompt, and"
            " {context} wherever the list of summaries to combine will be inserted."
        ),
    )
    # PLACEHOLDER. May later want to include... system_prompt: Annotated[str | None, Field(title="system prompt", description="Provides a role, context, and instructions for the LLM. This is a string with brackets around template variables. Must include {core_prompt}")] = None,
    max_concurrency: int = Field(
        default=3,
        title="Max concurrency",
        description=(
            "Maximum number of parallel calls to the LLM the summarizer is allowed to"
            " make."
        ),
    )
    iteration_limit: int = Field(
        default=3,
        title="Iteration limit",
        description=(
            "In a map-reduce summarization strategy, this is the maximum number of"
            ' times the LLM will "summarize the summaries".'
        ),
    )
    collapse_token_max: int = Field(
        default=6000,
        title="Collapse token max",
        description=(
            "In a map-reduce summarization strategy, this is the maximum number of"
            " tokens to include in the combined summaries that the LLM will summarize."
        ),
    )


class InputDocFormat(StrEnum):
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


async def sum_parser_selector(input_doc_format: InputDocFormat):
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


class SummarizationResult(BaseModel):
    status: str
    summary: str
    usage_report: str


@app.post("/summarize/{input_doc_format}", summary="Summarize a list of documents")
async def summarize(
    input_doc_format: InputDocFormat,
    docs_to_summarize: list[DocumentContents],
    preprocessor: Preprocessor,
    summarize_map_reduce: SummarizeMapReduce,
    api_key: str | None = Header(
        default=None,
        title="API key",
        description="API key for the summarization LLM. OpenAI is default",
    ),
) -> SummarizationResult:
    """
    Summarize a list of documents. Input doc format can be html, markdown, or text, but
    docs all must be of the same format.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pass an API key for the summarization LLM. OpenAI is default",
        )

    parser = await sum_parser_selector(input_doc_format)

    try:
        parsed_documents = await transform_raw_docs(
            docs_to_summarize,
            parser,
            preprocessor.max_tokens_per_doc,
            preprocessor.metadata_to_include,
        )

        if summarize_map_reduce.core_prompt is None:
            prompt = SummarizationTestPrompt.SIMPLE.value

        with get_openai_callback() as cb:
            summary = await map_reduce(
                parsed_documents,
                prompt,
                summarize_map_reduce.collapse_prompt,
                summarize_map_reduce.combine_prompt,
                max_concurrency=summarize_map_reduce.max_concurrency,
                iteration_limit=summarize_map_reduce.iteration_limit,
                collapse_token_max=summarize_map_reduce.collapse_token_max,
            )

            usage_report = cb

        return SummarizationResult(
            status="success",
            summary=summary,
            usage_report=repr(usage_report),
        )

    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error",
        )


@app.post("/summarize_from_disk")
async def summarize_from_disk(
    file_filter: FileFilter,
    preprocessor: Preprocessor,
    summarize_map_reduce: SummarizeMapReduce,
    api_key: str | None = Header(
        default=None,
        title="API key",
        description="API key for the summarization LLM. OpenAI is default",
    ),
) -> SummarizationResult:
    """
    Select and summarize a subset of files from a dataset on the server.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pass an API key for the summarization LLM. OpenAI is default",
        )

    try:
        input_files = filter_files(
            collection_digits=file_filter.collection_digits,
            dataset=DATASET_PATH,
            title_digits=file_filter.title_digits,
            file_format=DatasetFileFormatNames.HTML,
        )

        preprocessor: Coroutine[Any, Any, list[Document]] = transform_raw_docs(
            input_files,
            PARSE_FNS["markdownify_html_to_md"],
            preprocessor.max_tokens_per_doc,
            preprocessor.metadata_to_include,
        )

        parsed_documents = await preprocessor

        if summarize_map_reduce.core_prompt is None:
            prompt = SummarizationTestPrompt.SIMPLE.value

        with get_openai_callback() as cb:
            summary = await map_reduce(
                parsed_documents,
                prompt,
                summarize_map_reduce.collapse_prompt,
                summarize_map_reduce.combine_prompt,
                max_concurrency=summarize_map_reduce.max_concurrency,
                iteration_limit=summarize_map_reduce.iteration_limit,
                collapse_token_max=summarize_map_reduce.collapse_token_max,
            )
            usage_report = cb

        return SummarizationResult(
            status="success",
            summary=summary,
            usage_report=repr(usage_report),
        )

    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=50201, reload=True)
