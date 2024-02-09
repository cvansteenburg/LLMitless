import os
import tracemalloc
from enum import StrEnum
from logging import getLogger
from pathlib import Path
from typing import Callable, overload

import httpx
from fastapi import HTTPException, status
from langchain.callbacks import get_openai_callback
from pydantic import BaseModel

from src.chains.map_reduce import map_reduce
from src.models.chain_configs import (
    LLMConfigs,
    MapReduceConfigs,
    OpenAIFinishReason,
    PreprocessorConfig,
)
from src.services.doc_operations import (
    DocumentContents,
    SummarizationTestPrompt,
    transform_raw_docs,
)
from src.utils.callbacks import get_finish_reason_callback, get_log_error_callback

logger = getLogger(f"llmitless.{__name__}")

MEMCHECK = os.getenv("MEMCHECK", False)

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

        prompt = (
            summarize_map_reduce.core_prompt
            if summarize_map_reduce.core_prompt
            else SummarizationTestPrompt.PASSTHROUGH.value
        )

        with get_openai_callback() as cb:
            with get_finish_reason_callback() as finish_reason:
                with get_log_error_callback() as error_log:

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
                        if error_log.error_log:
                            logger.error(f"Error during chain run: {error_log.error_log}")
                        return SummarizationResult(
                            result_status=ResultStatus.SERVER_ERROR,
                            summary=None,
                            usage_report=None,
                            debug=None,
                        )
                    except Exception as general_err:
                        # Catch-all for other unexpected errors
                        logger.error(f"An unexpected error occurred: {general_err}")
                        if error_log.error_log:
                            logger.error(f"Error during chain run: {error_log.error_log}")
                        return SummarizationResult(
                            result_status=ResultStatus.SERVER_ERROR,
                            summary=None,
                            usage_report=None,
                            debug=None,
                        )
                    if error_log.error_log:
                        logger.error(f"Error during chain run: {error_log.error_log}")
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