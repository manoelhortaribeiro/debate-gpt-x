from typing import Optional

import pandas as pd

from debate_gpt.prompt_classes.openai_prompts.prompt_base import PromptBase


class WhoWon(PromptBase):
    def __init__(
        self,
        task_config: dict[str, str],
        debates_df: pd.DataFrame,
        rounds_df: pd.DataFrame,
        votes_df: pd.DataFrame,
        users_df: pd.DataFrame,
        big_issue_columns: Optional[list[str]] = None,
        demographic_columns: Optional[list[str]] = None,
        demographic_map: Optional[dict[str, str]] = None,
        max_gpt_response_tokens: Optional[int] = 1000,
        timeout: int = 120,
        source: str = "openai",
        model: str = "gpt-3.5-turbo-1106",
    ) -> None:
        super().__init__(
            debates_df=debates_df,
            rounds_df=rounds_df,
            votes_df=votes_df,
            users_df=users_df,
            big_issue_columns=big_issue_columns,
            demographic_columns=demographic_columns,
            demographic_map=demographic_map,
            voter_results=False,
            max_gpt_response_tokens=max_gpt_response_tokens,
            timeout=timeout,
            source=source,
            model=model,
        )

        self._task_config = task_config
        self._max_debate_tokens = self.calculate_max_debate_tokens()

    @property
    def max_debate_tokens(self):
        return self._max_debate_tokens

    def calculate_max_debate_tokens(self) -> int:
        base_token_count = sum(
            [
                self.count_tokens(self._task_config[config])
                for config in self._task_config.keys()
            ]
        )
        max_debate_tokens = (
            self.context_window - base_token_count - self.max_gpt_response_tokens
        )
        return max_debate_tokens

    def create_gpt_message(self, debate: str, debate_id: int) -> list[dict[str, str]]:
        message = [
            PromptBase.create_individual_gpt_message(
                "system", self._task_config["role_message"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["proposition_prefix"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", self.get_proposition(debate_id)
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["debate_prefix"]
            ),
            PromptBase.create_individual_gpt_message("user", debate),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["question"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["constraint"]
            ),
        ]
        return message
