import asyncio
import os
from typing import Annotated, Any, Coroutine

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, status
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from langchain.callbacks import get_openai_callback
from langchain_core.documents import Document
from pydantic import Field

from src.chains.map_reduce import map_reduce
from src.models.dataset_model import DatasetFileFormatNames
from src.parsers.html_parse import PARSE_FNS
from src.services.io import (
    DATASET_PATH,
    SummarizationTestPrompt,
    filter_files,
    html_to_md_documents,
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


app = FastAPI

CONFIG_FILE = "pyproject.toml"
logger = init_logging(CONFIG_FILE)

app.add_middleware(HTTPSRedirectMiddleware)

# Make sure we're live
@app.get("/")
async def root():
    logger.info("Hello World")
    return {"message": "Hello World"}

# Summarize from disk
# TODO: standardize language around core_prompt, make sure .format in map_reduce.py doesn't throw exception if there is not such template var in the input

# Summarize
# Specify format


@app.post("/summarize_from_disk")
async def summarize_from_disk(
    collection_digits: Annotated[str, Field(title="Collection digits", description='Usually a 3 digit number expressed as a string eg. "010"')],
    title_digits: Annotated[list[str], Field(title="Title digits", description='A list of usually 3 digit numbers expressed as a strings eg. ["001, "002", "009"]')],
    core_prompt: Annotated[str, Field(title="core prompt", description="Provides a role, context, and instructions for the LLM. LLM will summarize the content using this prompt. This is a string with brackets around template variables. At a minimum, prompt should include page_content in brackets like this: {page_content} where the contents of each summarized document will be inserted.")] = None,
    api_key: Annotated[str | None, Header(title="API key", description="Pass an API key for the summarization LLM. OpenAI is default")] = None,
    # PLACEHOLDER. May later want to include... system_prompt: Annotated[str | None, Field(title="system prompt", description="Provides a role, context, and instructions for the LLM. This is a string with brackets around template variables. Must include {core_prompt}")] = None,
    collapse_prompt: Annotated[str | None, Field(title="collapse prompt", description="In a map-reduce summarization strategy, the LLM will summarize the summaries to reduce the total amount of text. LLM will use this prompt to summarize the summaries. Use template variables {core_prompt} to include the core prompt, and {context} wherever the summaries to summarize will be inserted.")] = None,
    combine_summaries_prompt: Annotated[str | None, Field(title="combine summaries prompt", description="In a map-reduce summarization strategy, the last step is for the LLM to combine the summaries. LLM will use this prompt to combine the summaries. Use template variables {core_prompt} to include the core prompt, and {context} wherever the list of summaries to combine will be inserted.")] = None,
    *,
    max_tokens_per_doc: Annotated[int, Field(title="Max tokens per doc", description="The maximum number of tokens to include in each doc that the LLM will summarize. Change this to according the context window of the LLM and the length of the prompt.")] = 3000,
    iteration_limit: Annotated[int, Field(title="Iteration limit", description='In a map-reduce summarization strategy, this is the maximum number of times the LLM will "summarize the summaries".')] = 3,
    metadata_to_include: Annotated[list[str], Field(title="Metadata passed to LLM", description="In a map-reduce summarization strategy, docs are combined and presented together to the llm. The metadata keys are included in the combined documents to give the LLM more context.")]
) -> dict[str, str]:
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Pass an API key for the summarization LLM. OpenAI is default",
        )

    try:
        input_files = filter_files(
            collection_digits=collection_digits,
            dataset=DATASET_PATH,
            title_digits=title_digits,
            file_format=DatasetFileFormatNames.HTML,
        )

        preprocessor: Coroutine[Any, Any, list[Document]] = html_to_md_documents(
            input_files,
            PARSE_FNS["markdownify_html_to_md"],
            max_tokens_per_doc,
            metadata_to_include,
        )

        parsed_documents = await preprocessor

        if core_prompt is None:
            prompt = SummarizationTestPrompt.SIMPLE.value

        with get_openai_callback() as cb:
            summary = await map_reduce(parsed_documents, prompt)
            usage_report = cb

        return {
            "status": "success",
            "results": summary,
            "usage": usage_report
        }

    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=50201, reload=True)
