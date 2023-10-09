import os
from typing import Optional

import openai
import tiktoken
from openai.openai_object import OpenAIObject


class PromptBaseClass:
    def __init__(
        self,
        max_tokens: int = 4097,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        # set model info
        self._max_tokens = max_tokens
        self._model = model
        self._encoding = tiktoken.encoding_for_model(model)

        # set api key
        openai.api_key = os.environ["OPENAI_API_KEY"]

    @property
    def max_tokens(self):
        return self._max_tokens

    @staticmethod
    def get_user_message(message: str) -> dict[str, str]:
        return {"role": "user", "content": message}

    @staticmethod
    def get_system_message(message: str) -> dict[str, str]:
        return {"role": "system", "content": message}

    @staticmethod
    def get_assistant_message(message: str) -> dict[str, str]:
        return {"role": "assistant", "content": message}

    def count_tokens(self, message: str) -> int:
        """Return the number of tokens in `message` according to the encoding for the
        OpenAI model being used.
        """
        return len(self._encoding.encode(message))

    def prompt_chat_gpt(
        self, messages: list[str], max_tokens: Optional[int] = 16
    ) -> OpenAIObject:
        return openai.ChatCompletion.create(
            model=self._model, messages=messages, max_tokens=max_tokens
        )

    def calculate_cost_input(self, num_tokens: int) -> float:
        """Return the cost of inputting `num_tokens` into the model."""
        if self._model == "gpt-3.5-turbo":
            cost = (num_tokens * 0.0015) / 1000
        else:
            raise ValueError(f"Model {self._model} cost unknown.")

        return cost

    def calculate_cost_output(self, num_tokens: int) -> float:
        """Return the cost of outputting `num_tokens` from the model."""
        if self._model == "gpt-3.5-turbo":
            cost = (num_tokens * 0.002) / 1000
        else:
            raise ValueError(f"Model {self._model} cost unknown.")

        return cost
