import os
import tracemalloc
from contextlib import asynccontextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Callable, overload

import httpx
import sentry_sdk
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel

from src.chains.map_reduce import map_reduce
from src.models.chain_configs import (
    LLMConfigs,
    MapReduceConfigs,
    OpenAIFinishReason,
    PreprocessorConfig,
)
from src.models.dataset_model import (
    InputDocFormat,
    input_format_from_dataset_file_format,
    sum_parser_selector,
)
from src.services.io import (
    DocumentContents,
    SummarizationTestPrompt,
    transform_raw_docs,
)
from src.utils.callbacks import get_finish_reason_callback
from src.utils.client_auth import check_basic_auth
from src.utils.file_ops import FileFilter, filter_files
from src.utils.logging_init import init_logging

sentry_sdk.init(
    dsn="https://e6e80ca172e765ec75ad49a1137e2529@o4506601226764288.ingest.sentry.io/4506601237839872",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

ENV_CONTEXT = os.getenv("ENV_CONTEXT", "production")
MEMCHECK = os.getenv("MEMCHECK", False)

# Load the appropriate .env file
env_file = (
    ".env.prod"
    if ENV_CONTEXT == "production"
    else ".env.test" if ENV_CONTEXT == "test" else ".env"
)

load_dotenv(env_file)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    if MEMCHECK:
        tracemalloc.start()
    yield
    if MEMCHECK:
        tracemalloc.stop()


app = FastAPI(
    title="llmitless",
    description=(
        "Simple scaffolding, testbed, and API endpoints for building, testing, and"
        " deploying LLM chains."
    ),
    lifespan=app_lifespan,
)

CONFIG_FILE = "pyproject.toml"
logger = init_logging(CONFIG_FILE)

# if ENV_CONTEXT != "development":
#     app.add_middleware(HTTPSRedirectMiddleware)

CheckBasicAuth = Annotated[bool, Depends(check_basic_auth)]

LLMApiKey = Annotated[
    str,
    Header(
        ...,
        title="API key",
        description="API key for the LLM. Default LLM is OpenAI",
    ),
]


class ResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FORBIDDEN_CONTENT = "FORBIDDEN_CONTENT"
    UNCLEAR_INSTRUCTIONS = "UNCLEAR_INSTRUCTIONS"
    SERVER_ERROR = "MODEL_ERROR"


class SummarizationResult(BaseModel):
    result_status: ResultStatus
    summary: str | None = None
    usage_report: str | None = None
    debug: dict | None = None

    class Config:
        exclude_none = True


@overload
async def _summarize_sources(
    api_key: str,
    docs_to_summarize: list[DocumentContents],
    parser: Callable[[str], str],
    preprocessor_config: PreprocessorConfig,
    summarize_map_reduce: MapReduceConfigs,
    llm_config: LLMConfigs,
) -> SummarizationResult: ...
@overload
async def _summarize_sources(
    api_key: str,
    docs_to_summarize: list[Path],
    parser: Callable[[str], str],
    preprocessor_config: PreprocessorConfig,
    summarize_map_reduce: MapReduceConfigs,
    llm_config: LLMConfigs,
) -> SummarizationResult: ...
async def _summarize_sources(
    api_key: str,
    docs_to_summarize,
    parser: Callable[[str], str],
    preprocessor_config: PreprocessorConfig,
    summarize_map_reduce: MapReduceConfigs,
    llm_config: LLMConfigs,
) -> SummarizationResult:

    try:
        parsed_documents = await transform_raw_docs(
            docs_to_summarize,
            parser,
            preprocessor_config.max_tokens_per_doc,
            preprocessor_config.metadata_to_include,
        )

        if summarize_map_reduce.core_prompt is None:
            prompt = SummarizationTestPrompt.PASSTHROUGH.value

        with get_openai_callback() as cb:
            with get_finish_reason_callback() as finish_reason:

                try:
                    summary = await map_reduce(
                        parsed_documents,
                        prompt,
                        summarize_map_reduce.collapse_prompt,
                        summarize_map_reduce.combine_prompt,
                        api_key=api_key,
                        organization=llm_config.organization,
                        max_tokens=llm_config.max_tokens,
                        model=llm_config.model,
                        temperature=llm_config.temperature,
                        max_concurrency=summarize_map_reduce.max_concurrency,
                        iteration_limit=summarize_map_reduce.iteration_limit,
                        collapse_token_max=summarize_map_reduce.collapse_token_max,
                    )
                except httpx.HTTPError as http_err:
                    # Handle HTTP errors from any LLM provider
                    logger.error(f"HTTP error occurred: {http_err}")
                    return SummarizationResult(
                        result_status=ResultStatus.SERVER_ERROR,
                        summary=None,
                        usage_report=None,
                        debug=None,
                    )
                except Exception as general_err:
                    # Catch-all for other unexpected errors
                    logger.error(f"An unexpected error occurred: {general_err}")
                    return SummarizationResult(
                        result_status=ResultStatus.SERVER_ERROR,
                        summary=None,
                        usage_report=None,
                        debug=None,
                    )

                result_status = ResultStatus.SUCCESS
                usage_report = cb
                finish_reasons = finish_reason.finish_reasons

                if OpenAIFinishReason.CONTENT_FILTER.value in finish_reasons:
                    result_status = ResultStatus.FORBIDDEN_CONTENT
                elif OpenAIFinishReason.LENGTH.value in finish_reasons:
                    logger.warning("Summarization length limit reached.")

                debug_info = None
                if MEMCHECK:
                    current, peak = tracemalloc.get_traced_memory()
                    debug_info = {"current_memory": current, "peak_memory": peak}

                return SummarizationResult(
                    result_status=result_status,
                    summary=summary,
                    usage_report=repr(usage_report),
                    debug=debug_info,
                )

    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error",
        )


@app.get("/")
async def root():
    logger.info("Hello World")
    if MEMCHECK:
        current, peak = tracemalloc.get_traced_memory()
        return {"current_memory": current, "peak_memory": peak}
    return {"message": "Hello World"}


@app.post(
    "/summarize/{input_doc_format}",
    summary="Summarize a list of documents",
    operation_id="summarize",
    response_model_exclude_none=True,
)
async def summarize(
    api_key: LLMApiKey,
    input_doc_format: InputDocFormat,
    docs_to_summarize: list[DocumentContents],
    preprocessor: PreprocessorConfig,
    summarize_map_reduce: MapReduceConfigs,
    llm_config: LLMConfigs,
    auth: CheckBasicAuth,
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

    return await _summarize_sources(
        api_key,
        docs_to_summarize,
        parser,
        preprocessor,
        summarize_map_reduce,
        llm_config,
    )


@app.post(
    "/summarize_from_disk",
    operation_id="summarize_from_disk",
    response_model_exclude_none=True,
)
async def summarize_from_disk(
    api_key: LLMApiKey,
    file_filter: FileFilter,
    preprocessor_config: PreprocessorConfig,
    summarize_map_reduce: MapReduceConfigs,
    llm_config: LLMConfigs,
    auth: CheckBasicAuth,
) -> SummarizationResult:
    """
    Select and summarize a subset of files from a dataset on the server.
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pass an API key for the summarization LLM. OpenAI is default",
        )

    input_doc_format = input_format_from_dataset_file_format(file_filter.file_format)

    parser = await sum_parser_selector(input_doc_format)

    docs_to_summarize = filter_files(file_filter)

    return await _summarize_sources(
        api_key,
        docs_to_summarize,
        parser,
        preprocessor_config,
        summarize_map_reduce,
        llm_config,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app", host="127.0.0.1", port=50201, reload=True, log_level="trace"
    )
