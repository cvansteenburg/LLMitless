# LLMitless

### Build, test, and deploy LLM chains

Simple scaffolding, testbed, and API endpoints for building, testing, and deploying LLM chains.  
Supports local files, as well as those passed in directly or via API calls.

** consider experimental and unstable **

## Setup

### Devcontainer
For fastest setup, start the devcontainer in VS code.

### Local Setup
- Install dependencies via poetry.
- Add a `.env` file with your api keys (wandb, openai, etc)
- Use datasets_sample as a template for the hierarchy of local files for use with chains. Best numbered and laid out in the format shown in `datasets_sample`
- Set a port in `serve.py`, open the port (using ngrok, for example), and run `serve.py`.
- LLMitless uses FastAPI: once running, you can find documentation and make test calls by visiting /docs

### CI/CD
- Tests run automatically on pushes to main
- The manual Deploy action will dockerize the app, store the image in Google Artifact Registry, and deploy from there to Cloud Run.

## Key modules
`parsers/` contains modules that parse formats for use with LLMs

`services/summarize.py` collects docs, prepares them for use in a chain, and calls the LLM

`chains/` contains core langauge processing chains

`serve.py` Fast API endpoints
