import os
import tracemalloc
from contextlib import asynccontextmanager
from enum import Enum, StrEnum
from typing import Annotated

import httpx
import sentry_sdk
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, status
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel, Field

from src.chains.map_reduce import map_reduce
from src.models.dataset_model import (
    InputDocFormat,
    input_format_from_dataset_file_format,
    sum_parser_selector,
)
from src.services.io import (
    DocumentContents,
    FileFilter,
    SummarizationTestPrompt,
    filter_files,
    transform_raw_docs,
)
from src.utils.callbacks import get_finish_reason_callback
from src.utils.client_auth import check_basic_auth
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


# Make sure we're live
@app.get("/")
async def root():
    logger.info("Hello World")
    if MEMCHECK:
        current, peak = tracemalloc.get_traced_memory()
        return {"current_memory": current, "peak_memory": peak}
    return {"message": "Hello World"}


class PreprocessorConfig(BaseModel):
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


class MapReduceConfigs(BaseModel):
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
        default=3200,
        title="Collapse token max",
        description=(
            "In a map-reduce summarization strategy, this is the maximum number of"
            " tokens to include in the combined summaries that the LLM will summarize."
        ),
    )


class LLMConfigs(BaseModel):
    organization: str | None = Field(
        default=None,
        title="Organization",
        description=(
            "For users who belong to multiple organizations, you can pass a header"
            " to specify which organization is used for an API request. Usage from"
            " these API requests will count as usage for the specified"
            " organization."
        ),
    )
    model: str | None = Field(
        default=None,
        title="Model name",
        description=(
            "The model to use for LLM calls. If not specified, defaults to"
            " gpt-3.5-turbo"
        ),
    )
    temperature: float | None = Field(
        default=None,
        title="Temperature",
        description=(
            "Controls randomness of the output. Values closer to 0 make output more"
            " random, values closer to 1 make output more deterministic. If not"
            " specified, default is 0.7"
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        title="Max tokens",
        description=(
            "Maximum number of tokens the model will generate. If not specified,"
            " default is 3000"
        ),
    )


class ResultStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FORBIDDEN_CONTENT = "FORBIDDEN_CONTENT"
    UNCLEAR_INSTRUCTIONS = "UNCLEAR_INSTRUCTIONS"
    SERVER_ERROR = "MODEL_ERROR"


class OpenAIFinishReason(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


class OpenAIErrorCodes(Enum):
    INVALID_AUTHENTICATION = 401
    INCORRECT_API_KEY = 401
    RATE_LIMIT_REACHED = 429
    QUOTA_EXCEEDED = 429
    SERVER_ERROR = 500
    SERVER_OVERLOAD = 503


class SummarizationResult(BaseModel):
    result_status: ResultStatus
    summary: str | None = None
    usage_report: str | None = None
    debug: dict | None = None

    class Config:
        exclude_none = True


@app.post(
    "/summarize/{input_doc_format}",
    summary="Summarize a list of documents",
    operation_id="summarize", response_model_exclude_none=True,
)
async def summarize(
    api_key: Annotated[
        str,
        Header(
            ...,
            title="API key",
            description="API key for the LLM. Default LLM is OpenAI",
        ),
    ],
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

    try:
        parsed_documents = await transform_raw_docs(
            docs_to_summarize,
            parser,
            preprocessor.max_tokens_per_doc,
            preprocessor.metadata_to_include,
        )

        prompt = (
            summarize_map_reduce.core_prompt
            if summarize_map_reduce.core_prompt
            else SummarizationTestPrompt.SIMPLE.value
        )

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


@app.post("/summarize_from_disk", operation_id="summarize_from_disk", response_model_exclude_none=True)
async def summarize_from_disk(
    api_key: Annotated[
        str,
        Header(
            ...,
            title="API key",
            description="API key for the LLM. Default LLM is OpenAI",
        ),
    ],
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

    try:
        input_files = filter_files(file_filter)

        preprocessor = transform_raw_docs(
            input_files,
            parser,
            preprocessor_config.max_tokens_per_doc,
            preprocessor_config.metadata_to_include,
        )

        parsed_documents = await preprocessor

        if summarize_map_reduce.core_prompt is None:
            prompt = SummarizationTestPrompt.SIMPLE.value

        # TODO: Add callback manager and error handling here. Maybe call out HTTP errors that merit a retry from the client side. Get rid of JSON format in sample data and adjust tests.
        # TODO: Finish classes and handling in Briefly. Regenerate LLMitless client.
        with get_openai_callback() as cb:
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
            usage_report = cb

            debug_info = None

            if MEMCHECK:
                current, peak = tracemalloc.get_traced_memory()
                debug_info = {"current_memory": current, "peak_memory": peak}

        return SummarizationResult(
            result_status=ResultStatus.SUCCESS,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app", host="127.0.0.1", port=50201, reload=True, log_level="trace"
    )
