from enum import StrEnum


# list at tiktoken.model -> MODEL_TO_ENCODING
# https://platform.openai.com/docs/models/gpt-3-5
class ModelList(StrEnum):
    GPT4 = "gpt-4"
    GPT4_32K = "gpt-4-32k"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_16K = "gpt-3.5-turbo-16k"
    GPT_3_5_TURBO = "gpt-35-turbo" # to match tiktoken