from enum import Enum, StrEnum

from pydantic import BaseModel, Field


# list at tiktoken.model -> MODEL_TO_ENCODING
# https://platform.openai.com/docs/models/gpt-3-5
# https://platform.openai.com/docs/models/continuous-model-upgrades
class ModelList(StrEnum):
    GPT_3_5_TURBO_LATEST = "gpt-3.5-turbo"
    GPT_4_TURBO_LATEST = "gpt-4-turbo-preview"
    GPT_4_LATEST = "gpt-4"
    GPT4 = "gpt-4"
    GPT4_32K = "gpt-4-32k"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_TURBO = "gpt-35-turbo" # to match tiktoken


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