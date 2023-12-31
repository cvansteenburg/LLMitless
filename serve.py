import asyncio
import os
from typing import Any, Coroutine

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from langchain_core.documents import Document

from src.models.dataset_model import DatasetFileFormatNames
from src.parsers.html_parse import PARSE_FNS
from src.services.io import DATASET_PATH, filter_files, html_to_md_documents
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


@app.post("/summarize_from_files")
async def inbox(
    collection_digits: str,
    title_digits: list[str]
):
    MAX_TOKENS_PER_DOC = 3000
    ITERATION_LIMIT = 3
    METADATA_TO_INCLUDE = ["title"]  # metadata visible to llm in combined docs
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
            MAX_TOKENS_PER_DOC,
            METADATA_TO_INCLUDE,
        )

        parsed_documents = asyncio.run(preprocessor)

        return {
            "status": "success",
            "results": parsed_documents,
        }
    except Exception as e:
        # logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server error",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=50201, reload=True)
