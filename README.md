# LLMitless

### Build, test, and deploy LLM chains

Simple scaffolding, testbed, and API for building, testing, and deployment of LLM chains.  
Supports local files, as well as those passed in directly or via API calls.

** consider experimental and unstable **

## Setup
- Install dependencies via poetry.
- Add a `.env` file with your api keys (wandb, openai, etc)
- Use datasets_sample as a template for the hierarchy of local files for use with chains. Best numbered and laid out in the format shown in `datasets_sample`
- Set a port in `serve.py`, open the port (using ngrok, for example), and run `serve.py`.
- LLM_testbed uses FastAPI: once running, you can find documentation and make test calls by visiting /docs

## Key modules
`parsers/` contains modules that parse formats for use with LLMs

In `services`:
- `io.py` Collects docs, prepares them for use in a chain, and calls the LLM
- `output` Repository for output files

`chains/` contains various chains. Files with suffix _ex are examples for referencing when building chains.

`serve.py` Fast API endpoints
