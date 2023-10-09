import json
import os
import time
from typing import Union

import openai
import pandas as pd
import tqdm

from debate_gpt.prompt_classes.prompt_base_class import PromptBaseClass


class ContextQuestionConstraintClass(PromptBaseClass):
    def __init__(
        self,
        demographic_columns: list[str],
        users_df: pd.DataFrame,
        votes_df: pd.DataFrame,
        rounds_df: pd.DataFrame,
        debates_df: pd.DataFrame,
        max_tokens: int = 4097,
        model: str = "gpt-3.5-turbo",
    ) -> None:
        """This class is responsible for holding all the methods related to prompting
        ChatGPT for the following task: Given a debate and a user's demographic data,
        which side of the debate is the user most likely to agree with?"""
        super().__init__(max_tokens, model)

        # set dataframes
        self._users_df = users_df
        self._debates_df = debates_df
        self._votes_df = votes_df
        self._rounds_df = rounds_df

        self._task_structure = {
            "context": (
                "Consider the following debate which contains rounds of a 'pro' and "
                "'con' argument:"
            ),
            "question": (
                "Given the debate, which side would a person with the following "
                "demographics vote for? Demographic Information:"
            ),
            "constraint": (
                "Even if you are uncertain, you must pick either 'Pro', 'Con', or 'Tie'"
                " without using any other words or punctuation."
            ),
        }
        self._demographic_columns = demographic_columns

        # determine the amount of tokens used in the prompt boiler plate
        self._base_token_count = sum(
            [
                self.count_tokens(self._task_structure[key])
                for key in self._task_structure.keys()
            ]
        )
        # calculate the maximum possible tokens used for user demographic information
        self._max_tokens_demographics = self.calculate_max_tokens_demographics()

        self._max_gpt_tokens = 2

        # calculate the maximum number of tokens a debate can be
        self._max_tokens_debate = (
            self.max_tokens
            - self._base_token_count
            - self._max_tokens_demographics
            - self._max_gpt_tokens
        )

        self.filter_debates()

    @property
    def votes_df(self):
        return self._votes_df

    @property
    def users_df(self):
        return self._users_df

    @property
    def rounds_df(self):
        return self._rounds_df

    @property
    def debates_df(self):
        return self._debates_df

    def calculate_max_tokens_demographics(self) -> int:
        """Return the maximum number of tokens that may be used in user demographics."""
        message = []
        for col in self._demographic_columns:
            possible_column_values = list(self._users_df[col].unique())
            max_count = 0
            max_value = None
            for column_value in possible_column_values:
                if column_value is None:
                    continue
                token_count = self.count_tokens(column_value)
                if token_count > max_count:
                    max_count = token_count
                    max_value = column_value

            message.append(col.replace("_", " ").title() + ": " + max_value)

        return self.count_tokens(", ".join(message))

    def filter_debates(self) -> None:
        self._rounds_df["cum_sum"] = self._rounds_df.cum_sum + (
            4 * self._rounds_df.order
        )
        self._rounds_df = self._rounds_df[
            self._rounds_df.cum_sum <= self._max_tokens_debate
        ]

    def get_demographics(self, voter_id: str) -> str:
        """Return a string containing all the demographic information of user `voter_id`
        in the following format: Label: Value, Label: Value
        """
        voter_demographics = self.users_df.loc[voter_id][self._demographic_columns]
        voter_demographics = voter_demographics[~voter_demographics.isna()]
        voter_demographics = [
            col.replace("_", " ").title() + ": " + voter_demographics[col]
            for col in voter_demographics.index
        ]

        return ", ".join(voter_demographics)

    def get_vote(self, voter_id: str, debate_id: int) -> str:
        """Return either 'Pro', 'Con' or 'Tie' indicating how user `voter_id` voted on
        debate `debate_id`.
        """
        row = self.votes_df[
            (self.votes_df.debate_id == debate_id)
            & (self.votes_df.voter_id == voter_id)
        ]
        if row.agreed_after.values[0] == row.pro_user_id.values[0]:
            return "Pro"
        elif row.agreed_after.values[0] == row.con_user_id.values[0]:
            return "Con"
        else:
            return "Tie"

    def get_debate(self, debate_id: int) -> str:
        debate = self.rounds_df[self.rounds_df.debate_id == debate_id]
        max_order = debate.order.max()

        if (max_order % 2) != 0:
            max_order = max_order - 1

        rounds = []
        for _, row in debate.iterrows():
            rounds.append(row.side + ": " + row.text + " ")
        return " ".join(rounds)

    def create_message_context_question_constraint(
        self, debate: str, user_demographics: str
    ) -> list[dict[str, str]]:
        message = [
            PromptBaseClass.get_user_message(self._task_structure["context"]),
            PromptBaseClass.get_user_message(debate),
            PromptBaseClass.get_user_message(self._task_structure["question"]),
            PromptBaseClass.get_user_message(user_demographics),
            PromptBaseClass.get_user_message(self._task_structure["constraint"]),
        ]
        return message

    def get_debate_predictions(self, debate_id: int) -> list[dict[str, str]]:
        results = []

        debate = self.get_debate(debate_id)
        voter_ids = list(self.votes_df[self.votes_df.debate_id == debate_id].voter_id)
        for voter_id in voter_ids:
            demographics = self.get_demographics(voter_id)
            vote = self.get_vote(voter_id, debate_id)

            message = self.create_message_context_question_constraint(
                debate, demographics
            )
            try:
                response = self.prompt_chat_gpt(
                    message, max_tokens=self._max_gpt_tokens
                )
                results.append(
                    {
                        "debate_id": int(debate_id),
                        "voter_id": voter_id,
                        "actual_vote": vote,
                        "gpt_prediction": response["choices"][0]["message"]["content"],
                    }
                )
            except openai.error.APIError as e:
                print(e)
                return None
            except openai.error.ServiceUnavailableError as e:
                print(e)
                return None
            except Exception as e:
                print(e)
                return None

        return results

    @staticmethod
    def save_results_to_file(
        results: list[dict[str, Union[int, str]]], path_to_file: str
    ):
        if os.path.isfile(path_to_file):
            with open(path_to_file) as f:
                results_old = json.load(f)
                results = results_old + results

        with open(path_to_file, "w") as f:
            json.dump(results, f)

    def get_batch_predictions(self, debate_ids: list[int], path_to_file: str) -> None:
        prompted_debate_ids = []
        while len(debate_ids) != 0:
            results = []
            for debate_id in tqdm.tqdm(debate_ids):
                results_tmp = self.get_debate_predictions(debate_id)
                if results_tmp is None:
                    break

                # TODO: why would this happen?
                if len(results) == 0:
                    continue

                prompted_debate_ids.append(debate_id)
                results += results_tmp
                time.sleep(1)

            debate_ids = [
                item for item in debate_ids if item not in prompted_debate_ids
            ]

            ContextQuestionConstraintClass.save_results_to_file(results, path_to_file)
