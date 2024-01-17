from typing import Optional

import numpy as np
import pandas as pd

from debate_gpt.prompt_classes.prompt_base import PromptBase


class PropositionVoter(PromptBase):
    def __init__(
        self,
        task_config: dict[str, str],
        debates_df: pd.DataFrame,
        rounds_df: pd.DataFrame,
        votes_df: pd.DataFrame,
        users_df: pd.DataFrame,
        big_issue_columns: Optional[list[str]],
        demographic_columns: Optional[list[str]],
        demographic_map: Optional[dict[str, str]],
        max_gpt_response_tokens: Optional[int] = 2,
        source: str = "openai",
        model: str = "gpt-3.5-turbo",
    ) -> None:
        super().__init__(
            debates_df,
            rounds_df,
            votes_df,
            users_df,
            big_issue_columns=big_issue_columns,
            demographic_columns=demographic_columns,
            demographic_map=demographic_map,
            voter_results=True,
            max_gpt_response_tokens=max_gpt_response_tokens,
            source=source,
            model=model,
        )

        self._task_config = task_config

    def calculate_max_debate_tokens(self) -> int:
        pass

    @property
    def max_debate_tokens(self):
        return np.inf

    def create_gpt_message(
        self, debate: str, debate_id: int, voter_id: str
    ) -> list[dict[str, str]]:
        role_message = self.create_role_text(voter_id, debate_id)
        if role_message == "":
            return None
        message = [
            PromptBase.create_individual_gpt_message("system", role_message),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["context"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", self.get_proposition(debate_id)
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["question"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["constraint"]
            ),
        ]
        return message
