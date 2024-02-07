from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator

from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult

finish_reason_callback_var: ContextVar[GetFinishReason | None] = ContextVar(
    "finish_reason_callback", default=None
)

class GetFinishReason(BaseCallbackHandler):
    """
    A callback handler that collects 'finish_reason' from each generation in a response.
    
    This handler is designed to be used as a callback at the end of an LLM operation to
    aggregate the reasons why the LLM operation was concluded.

    Attributes:
        finish_reasons: A list that stores the 'finish_reason' values from the generations.
    """

    finish_reasons: list[str] = []

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Executed when the LLM operation concludes, extracting 'finish_reason' from
        each generation's information.

        If 'finish_reason' is present within a generation's information, it is appended
        to the 'finish_reasons' list attribute of this class.

        Args:
            response: An LLMResult object containing the results of an LLM operation.
            **kwargs: Additional keyword arguments that might be provided, but are not used here.
        """
        for generation_list in response.generations:
            for generation in generation_list:
                finish_reason = (
                    generation.generation_info.get("finish_reason", None)
                    if generation.generation_info
                    else None
                )
                if finish_reason:
                    self.finish_reasons.append(finish_reason)


@contextmanager
def get_finish_reason_callback() -> Generator[GetFinishReason, None, None]:
    """
    A context manager to instantiate and manage the lifecycle of a GetFinishReason object.

    It sets a GetFinishReason instance as the current context for collecting 'finish_reason'
    values during an LLM operation and ensures the context is cleared after use.

    Yields:
        An instance of GetFinishReason to accumulate 'finish_reason' within a given context.
    """
    finish_info = GetFinishReason()
    finish_reason_callback_var.set(finish_info)
    yield finish_info
    finish_reason_callback_var.set(None)