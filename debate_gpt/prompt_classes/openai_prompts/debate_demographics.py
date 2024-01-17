from typing import Optional

import pandas as pd

from debate_gpt.prompt_classes.openai_prompts.prompt_base import PromptBase


class DebateDemographics(PromptBase):
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
        max_gpt_response_tokens: Optional[int] = 50,
        timeout: int = 120,
        source: str = "openai",
        model: str = "gpt-3.5-turbo-1106",
    ) -> None:
        """This class is responsible for holding all the methods related to prompting
        ChatGPT for the following task: Given a debate and a user's demographic data,
        which side of the debate is the user most likely to agree with?
        """
        super().__init__(
            debates_df=debates_df,
            rounds_df=rounds_df,
            votes_df=votes_df,
            users_df=users_df,
            big_issue_columns=big_issue_columns,
            demographic_columns=demographic_columns,
            demographic_map=demographic_map,
            voter_results=True,
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
        return (
            self.context_window
            - base_token_count
            - super().calculate_max_user_info_tokens()
            - (2 * self.max_gpt_response_tokens)
        )

    def create_gpt_message(
        self, debate: str, debate_id: str, voter_id: str
    ) -> list[dict[str, str]]:
        """Return the message that will be sent to prompt the LLM. This is in the format
        of context, question, constraint."""
        message = [
            PromptBase.create_individual_gpt_message(
                "system", self._task_config["role_message"]
            ),
            PromptBase.create_individual_gpt_message(
                "system", self.create_date_cutoff_role_text(debate_id)
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["proposition_prefix"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", super().get_proposition(debate_id)
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["debate_prefix"]
            ),
            PromptBase.create_individual_gpt_message("user", debate),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["user_demographics_prefix"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", super().get_user_info(voter_id)
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["question"]
            ),
            PromptBase.create_individual_gpt_message(
                "user", self._task_config["constraint"]
            ),
        ]
        return message
