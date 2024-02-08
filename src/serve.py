import os
import tracemalloc
from contextlib import asynccontextmanager
from typing import Annotated

import sentry_sdk
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status

from src.models.chain_configs import (
    LLMConfigs,
    MapReduceConfigs,
    PreprocessorConfig,
)
from src.models.dataset_model import (
    InputDocFormat,
    input_format_from_dataset_file_format,
    sum_parser_selector,
)
from src.services.doc_operations import DocumentContents
from src.services.summarize import SummarizationResult, _summarize_sources
from src.utils.client_auth import check_basic_auth
from src.utils.file_operations import FileFilter, filter_files
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
