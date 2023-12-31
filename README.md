# LLM Testbed
** consider experimental and unstable **

Basic scaffolding / testbed for building LLM chains and testing.
Supports local files, as well as those passed in directly or via API calls.

## Setup
- Intended to be run locally. Install dependencies via poetry.
- Add a '.env' file with api keys (wandb, openai, etc)
- Use datasets_sample as a template for the hierarchy of local files for use with chains. Best numbered and laid out in the format shown in `datasets_sample`

## Key modules
`parsers/` contains modules that parse formats for use with LLMs

In `services`:
- `io.py` Collects docs, prepares them for use in a chain, and calls the LLM
- `output` Repository for output files

`chains/` contains various chains. Files with suffix _ex are examples for referencing when building chains.

`serve.py` Langsmith (FastAPI) for generating chain access endpoints
