from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langserve import add_routes  # type: ignore

from src.chains.list_example import category_chain

# from src.chains.stuff_ex import stuff_chain

app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

add_routes(
    app,
    category_chain,
    path="/category_chain",
)

# add_routes(
#     app,
#     stuff_chain,
#     path="/stuff_chain",
# )

add_routes(
    app,
    ChatOpenAI(),
    path="/openai",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)