import os
import tracemalloc
from contextlib import asynccontextmanager
from typing import Annotated

import sentry_sdk
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, status

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
from src.services.summarize import ResultStatus, SummarizationResult, _summarize_sources
from src.utils.client_auth import check_basic_auth
from src.utils.file_operations import FileFilter, filter_files
from src.utils.logging_init import init_logging

sentry_sdk.init(
    dsn="https://e6e80ca172e765ec75ad49a1137e2529@o4506601226764288.ingest.sentry.io/4506601237839872",
    # Set to 1.0 to capture 100% of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set to 1.0 to profile 100% of sampled transactions
    profiles_sample_rate=1.0,
)

ENV_CONTEXT = os.getenv("ENV_CONTEXT", "production")

env_file = (
    ".env.prod"
    if ENV_CONTEXT == "production"
    else ".env.test" if ENV_CONTEXT == "test" else ".env"
)

load_dotenv(env_file)

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    memcheck_status = os.getenv("MEMCHECK", False)
    if memcheck_status:
        tracemalloc.start()
    yield
    if memcheck_status:
        tracemalloc.stop()


app = FastAPI(
    title="llmitless",
    description=(
        "Simple scaffolding, testbed, and API endpoints for building, testing, and"
        " deploying LLM chains."
    ),
    lifespan=app_lifespan,
)

logger = init_logging("pyproject.toml")


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

# Create a header that allows me to toggle 


@app.get("/")
async def root():
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


# Test endpoint mirror's "summarize" endpoint signature, returns deterministic SummarizationResult if well-formed "summarize" parameters are received
@app.post(
    "/summarize_test",
    operation_id="summarize_test",
    response_model_exclude_none=True,
    description="Test endpoint that mirrors the 'summarize' endpoint signature, returns deterministic SummarizationResult if well-formed 'summarize' parameters are received.",
)
async def summarize_test(
    api_key: LLMApiKey,
    input_doc_format: InputDocFormat,
    docs_to_summarize: list[DocumentContents],
    preprocessor: PreprocessorConfig,
    summarize_map_reduce: MapReduceConfigs,
    llm_config: LLMConfigs,
    auth: CheckBasicAuth,
    reqeust: Request,
    test_scenario: str | None = Query(None),
) -> SummarizationResult:
    """
    Simulates various outcomes of a summarization request based on the test scenario provided.

    This endpoint is designed for testing client-side handling of different API responses. By specifying a `test_scenario` query parameter, you can simulate various outcomes, such as successful summarization, server errors, or authentication failures. This allows client developers to test their code against predictable responses without triggering actual backend logic.

    Parameters:
    - test_scenario: Optional query parameter to specify the simulation scenario.

    Returns:
    - SummarizationResult: The simulated result of the summarization request, including status and optionally summary, usage report, and debug information.

    Available scenarios include:
    - "INAUTHENTICATED": Simulates an authentication failure response (note: this scenario is theoretical and might not be triggered as described).
    - "INVALID_REQUEST": Simulates a response for an invalid request (note: like 'INAUTHENTICATED', this scenario is theoretical and might not behave as described due to FastAPI's validation mechanisms).
    - "FORBIDDEN_CONTENT": Simulates a response for forbidden content.
    - "SERVER_ERROR": Simulates a server error response.
    - "SUCCESS_WITH_DUMMY_DATA": Returns a successful response with summary and dummy data for other fields.
    - "SUCCESS_WITH_NONE": Returns a successful response with no summary or other data.
    - "SUCCESS_WITH_SUMMARY_ONLY": Returns a successful response with a summary.
    """
    match test_scenario:
        case "INAUTHENTICATED":
            logger.error(f"INAUTHENTICATED TEST SUCCEEDED when it should have failed with request: {reqeust}")
            return SummarizationResult(
                result_status=ResultStatus.SUCCESS,
                summary="WARNING - AUTH PASSED WHEN IT SHOULD HAVE FAILED."
            )
        case "INVALID_REQUEST":
            logger.error(f"INVALID_REQUEST TEST SUCCEEDED when it should have failed with request: {reqeust}")
            return SummarizationResult(
                result_status=ResultStatus.SUCCESS,
                summary="WARNING - REQUEST PASSED VALIDATION WHEN IT SHOULD HAVE FAILED."
            )
        case "FORBIDDEN_CONTENT":
            return SummarizationResult(
                result_status=ResultStatus.FORBIDDEN_CONTENT,
                summary="Successful summarization output"
            )
        case "SERVER_ERROR":
            return SummarizationResult(
                result_status=ResultStatus.SERVER_ERROR
            )
        case "SUCCESS_WITH_DUMMY_DATA":
            return SummarizationResult(
                result_status=ResultStatus.SUCCESS,
                summary="Successful summarization output",
                usage_report="Dummy usage report",
                debug={"key": "value"}
            )
        case "SUCCESS_WITH_NONE" | None:
            return SummarizationResult(
                result_status=ResultStatus.SUCCESS
            )
        case "SUCCESS_WITH_SUMMARY_ONLY":
            return SummarizationResult(
                result_status=ResultStatus.SUCCESS,
                summary="Successful summarization output"
            )
        case _:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid test_scenario: {test_scenario}",
            )
# check each argument matches the expected type. If so, return a SummarizationResult. Otherwise, raise an exception. Do not call the summarization chain (we're just checking input and returning an output for the client integration test to handle).



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app", host="127.0.0.1", port=50201, reload=True, log_level="trace"
    )