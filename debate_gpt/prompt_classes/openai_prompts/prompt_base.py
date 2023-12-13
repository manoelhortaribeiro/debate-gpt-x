import datetime
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import openai
import pandas as pd
import tiktoken
import tqdm


class PromptBase(ABC):
    def __init__(
        self,
        debates_df: pd.DataFrame,
        rounds_df: Optional[pd.DataFrame],
        votes_df: Optional[pd.DataFrame],
        users_df: Optional[pd.DataFrame],
        big_issue_columns: Optional[list[str]],
        demographic_columns: Optional[list[str]],
        demographic_map: Optional[dict[str, str]],
        voter_results: bool = True,
        max_gpt_response_tokens: Optional[int] = 2,
        timeout: int = 120,
        source: str = "openai",
        model: str = "gpt-3.5-turbo-1106",
    ) -> None:
        """This is the abstract base class for all prompting of OpenAI models for the
        debate-gpt project.

        Data:
        `debates_df`, `rounds_df`, `votes_df`, and `users_df` are all pandas dataframes
        containing the data related to this project. `debates_df` contains all debate
        meta data including: debate_id, start_date, pro_user_id, con_user_id, and
        category.

        OpenAI model parameters:
        `model` defines the OpenAI model to be used and can be one of the options found
        here: https://platform.openai.com/docs/models. `context_window` should be the
        corresponding context window found on the same page.
        """

        # set dataframes
        self._users_df = users_df
        self._debates_df = debates_df
        self._votes_df = votes_df
        self._rounds_df = rounds_df

        # set model info
        self._source = source
        self._model = model
        self._context_window = self.get_model_context_window()
        self._timeout = timeout
        if (model == "llama") | (model == "mistral"):
            self._encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        else:
            self._encoding = tiktoken.encoding_for_model(model)

        self._max_gpt_response_tokens = max_gpt_response_tokens

        # set api key
        if source == "openai":
            openai.api_key = os.environ["OPENAI_API_KEY"]
        else:
            self._client = openai.OpenAI(
                api_key="anything",
                base_url="http://iccluster039.iccluster.epfl.ch:7736",
            )

        # set user columns
        self._voter_results = voter_results
        self._big_issue_columns = big_issue_columns
        self._demographic_columns = demographic_columns
        self._demographic_map = demographic_map
        if self._demographic_map is not None:
            assert self._demographic_columns is not None

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

    @debates_df.setter
    def debates_df(self, new_debates_df):
        self._debates_df = new_debates_df

    @property
    def context_window(self):
        return self._context_window

    @property
    def max_gpt_response_tokens(self):
        return self._max_gpt_response_tokens

    @property
    @abstractmethod
    def max_debate_tokens(self):
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def create_gpt_message(self):
        raise NotImplementedError("This is an abstract method.")

    @abstractmethod
    def calculate_max_debate_tokens(self) -> int:
        raise NotImplementedError("This is an abstract method.")

    def get_batch_results(self, debate_ids: list[int], path_to_file: str):
        """Get the results of prompting the model for all debates in `debate_ids` and
        save in `path_to_file`.
        """
        results = []  # initalize results list
        for i, debate_id in enumerate(tqdm.tqdm(debate_ids)):
            curr_debate_results = self.get_results(debate_id)

            if len(curr_debate_results) == 0:
                continue

            results += curr_debate_results
            if len(results) >= 50:
                self.save_results_to_file(results, path_to_file)
                results = []

            time.sleep(0.5)

        self.save_results_to_file(results, path_to_file)

    def get_results(self, debate_id: int):
        """Return results from prompting the model for debate with id `debate_id`."""
        if self._voter_results:
            return self.get_voter_debate_results(debate_id)
        else:
            return self.get_debate_results(debate_id)

    def get_debate_results(self, debate_id: str):
        """Return results from prompting the model for debate with id `debate_id` with
        no user personalization."""
        results = []
        debate = self.get_debate(debate_id=debate_id)
        if debate == "":
            return results

        message = self.create_gpt_message(debate, debate_id)
        response = None
        while response is None:
            try:
                response = self.prompt(message, self.max_gpt_response_tokens)
            except Exception as e:
                print(e)

        results.append(
            {
                "debate_id": debate_id,
                "message": message,
                "gpt_response": response.choices[0].message.content,
            }
        )

        return results

    def get_voter_debate_results(self, debate_id: str):
        """Return results from prompting the model for debate with id `debate_id` with
        voter level personalization.
        """
        results = []
        debate = self.get_debate(debate_id=debate_id)
        if debate == "":
            return results

        voter_ids = list(self.votes_df[self.votes_df.debate_id == debate_id].voter_id)

        while len(voter_ids) != 0:
            flag = False
            for i, voter_id in enumerate(voter_ids):
                message = self.create_gpt_message(debate, debate_id, voter_id)

                if message is None:
                    continue

                try:
                    response = self.prompt(message, self.max_gpt_response_tokens)
                except Exception as e:
                    print(e)
                    flag = True
                    break

                results.append(
                    {
                        "debate_id": debate_id,
                        "voter_id": voter_id,
                        "message": message,
                        "gpt_response": response.choices[0].message.content,
                        "agreed_before": self.get_column_vote(
                            voter_id, debate_id, "agreed_before"
                        ),
                        "agreed_after": self.get_column_vote(
                            voter_id, debate_id, "agreed_after"
                        ),
                    }
                )

            if not flag:
                i += 1
            voter_ids = voter_ids[i:]

        return results

    def get_proposition(self, debate_id: str) -> str:
        """Return the proposition associated with the debate with id `debate_id`."""
        return (
            self.debates_df[self.debates_df.debate_id == debate_id]
            .proposition.values[0]
            .capitalize()
        )

    def get_debate(self, debate_id: int) -> str:
        """Return the debate with id `debate_id`.

        The format of the debate is as follows:
        {Round 0: {
            Pro: pro argument,
            Con: con argument,
            },
         Round 1: {
            Pro: pro argument,
            Con: con argument,
            },
         etc.
        }
        """
        rounds = {}
        debate_df = self.rounds_df[self.rounds_df.debate_id == debate_id]

        for _, row in debate_df.iterrows():
            round_key = f"Round {row['round']}"
            if round_key in rounds.keys():
                old_dict = rounds[round_key]
                old_dict[row.side] = row.text
                rounds[round_key] = old_dict
            else:
                rounds[round_key] = {row.side: row.text}

            if self.count_tokens(str(json.dumps(rounds))) >= self.max_debate_tokens:
                break

        debate = str(json.dumps(rounds))
        if self.count_tokens(debate) <= self.max_debate_tokens:
            return debate

        del rounds[round_key]
        return str(json.dumps(rounds))

    def get_column_vote(self, voter_id: str, debate_id: int, column: str) -> str:
        """Return either 'Pro', 'Con' or 'Tie' indicating how user `voter_id` voted on
        debate `debate_id` for `column`.
        """
        row = self.votes_df[
            (self.votes_df.debate_id == debate_id)
            & (self.votes_df.voter_id == voter_id)
        ]
        return row[column].values[0]

    def create_demographics_role_text(self, voter_id: str) -> str:
        """Craft system role text for demographic information for user `voter_id`."""
        message = ""
        if self._demographic_columns is None:
            return message

        user = self.users_df.loc[voter_id]
        for demographic in self._demographic_columns:
            if user[demographic]:
                message += (
                    self._demographic_map[demographic]
                    + user[demographic].lower()
                    + ". "
                )

        return message

    def create_big_issues_role_text(self, voter_id: str) -> str:
        """Craft a system role text for big issues for user `voter_id`."""
        message = ""
        if self._big_issue_columns is None:
            return message
        pro_issues = []
        con_issues = []
        undecided_issues = []

        for big_issue in self._big_issue_columns:
            big_issue_name = big_issue.replace("_", " ").title()
            if self.users_df.loc[voter_id][big_issue] == "Pro":
                pro_issues.append(big_issue_name)
            elif self.users_df.loc[voter_id][big_issue] == "Con":
                con_issues.append(big_issue_name)
            elif self.users_df.loc[voter_id][big_issue] == "Und":
                undecided_issues.append(big_issue_name)

        if len(pro_issues) > 0:
            message += f"You are for the following issues: {', '.join(pro_issues)}. "
        if len(con_issues) > 0:
            message += (
                f"You are against the following issues: {', '.join(con_issues)}. "
            )
        if len(undecided_issues) > 0:
            message += "You are undecided about the following issues: "
            message += f"{', '.join(undecided_issues)}. "

        return message

    def create_date_cutoff_role_text(self, debate_id) -> str:
        date = (
            self.debates_df[self.debates_df.debate_id == debate_id]
            .iloc[0]
            .at["start_date"]
        )
        date = date.replace("\\", "")
        date = datetime.datetime.strptime(date, "%m/%d/%Y")
        date = date.strftime("%B %d, %Y")

        message = f"The date is {date}. "
        message += (
            "You have no information on any events that happened after this date. "
        )
        message += "You have no access to information released after this date."
        return message

    def create_role_text(self, voter_id: str, debate_id: int) -> str:
        """Craft system role text for debate with id `debate_id` and for user
        `voter_id`.
        """
        message = ""
        message += self.create_demographics_role_text(voter_id)
        message += self.create_big_issues_role_text(voter_id)
        message += self.create_date_cutoff_role_text(debate_id=debate_id)
        return message

    def get_user_info(self, voter_id: str) -> str:
        """Return a string containing all the demographic information of user `voter_id`
        in the following format: Label: Value, Label: Value
        """
        voter_info = self.users_df.loc[voter_id][self._demographic_columns]
        voter_info = voter_info[~voter_info.isna()]
        voter_info = [
            col.replace("_", " ").title() + ": " + voter_info[col]
            for col in voter_info.index
        ]

        return "\n".join(voter_info)

    def _calculate_max_role_tokens(self) -> int:
        message = ""
        if self._demographic_columns is not None:
            for demographic in self._demographic_columns:
                possible_values = list(self.users_df[demographic].unique())
                max_count = 0
                max_value = None
                for value in possible_values:
                    if value is None:
                        continue
                    token_count = self.count_tokens(value)
                    if token_count > max_count:
                        max_count = token_count
                        max_value = value

                message += self._demographic_map[demographic] + max_value.lower() + ". "

        if self._big_issue_columns is not None:
            big_issues = [
                issue.replace("_", "").title() for issue in self._big_issue_columns
            ]
            message += f"You are for the following issues: {', '.join(big_issues[0])}. "
            message += (
                f"You are against the following issues: {', '.join(big_issues[1])}. "
            )
            message += "You are undecided about the following issues: "
            message += f"{', '.join(big_issues[2])}. "
            message += "You have no opinion about the following issues: "
            message += f"{', '.join(big_issues[3:])}. "
        return self.count_tokens(message)

    def calculate_max_user_info_tokens(self) -> int:
        """Return the maximum number of tokens that may be used in user demographics."""
        message = []
        for col in self._demographic_columns:
            possible_column_values = list(self.users_df[col].unique())
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

        return self.count_tokens("\n".join(message))

    @staticmethod
    def save_results_to_file(results: list[dict[str, str]], path_to_file: str) -> None:
        """Save `results` into `path_to_file`.

        The file should be a JSON file. If the file already exists, the results will be
        appended to the current contents of the file. Otherwise, a new file will be
        created.
        """
        if os.path.isfile(path_to_file):
            with open(path_to_file) as f:
                results_old = json.load(f)
                results = results_old + results

        with open(path_to_file, "w") as f:
            json.dump(results, f)

    @staticmethod
    def create_individual_gpt_message(role: str, message: str) -> dict[str, str]:
        """Create a message object for prompting the model as described here:
        https://platform.openai.com/docs/guides/text-generation/chat-completions-api.
        """
        if role not in ["user", "system", "assistant"]:
            raise ValueError(
                f"{role} not a valid role. Try 'user', 'system', or 'assistant'."
            )
        return {"role": role, "content": message}

    def prompt(self, messages, max_tokens):
        if self._source == "openai":
            return self.prompt_chat_gpt(messages, max_tokens)
        else:
            return self.prompt_open_source_model(messages, max_tokens)

    def prompt_open_source_model(
        self, messages: list[str], max_tokens: Optional[int] = 50
    ):
        return self._client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages
        )

    def prompt_chat_gpt(self, messages: list[str], max_tokens: Optional[int] = 50):
        """Prompt the OpenAI model and return the chat completions object."""
        return openai.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            request_timeout=self._timeout,
        )

    def count_tokens(self, message: str) -> int:
        """Return the number of tokens in `message` according to the encoding for the
        OpenAI model being used.
        """
        return len(self._encoding.encode(message))

    def get_model_context_window(self) -> int:
        """Return the context window of the model in use. These can be found at
        https://platform.openai.com/docs/models and should be updated regularly should
        the context windows change.

        Last update: Nov 29, 2023.
        """
        if self._model == "gpt-3.5-turbo-1106":
            return 16385
        elif (
            (self._model == "gpt-3.5-turbo-0613")
            | (self._model == "gpt-3.5-turbo")
            | (self._model == "llama")
            | (self._model == "mistral")
        ):
            # gpt-3.5-turbo currently points to gpt-3.5-turbo-0613. Will point to
            # gpt-3.5-turbo-1106 starting Dec 11, 2023
            return 4096
        elif self._model == "gpt-4":
            return 8192
        elif self._model == "gpt-4-32k":
            return 32768
        else:
            raise ValueError(f"Context window unknown for model {self._model}.")

    def calculate_cost_input(self, num_tokens: int) -> float:
        """Return the cost of inputting `num_tokens` into the model. This should be
        updated regularly according to https://openai.com/pricing. As new models are
        released, their pricing information should be added to this function.

        Last update: Nov 29, 2023.
        """
        if (self._model == "gpt-3.5-turbo-1106") | (self._model == "gpt-3.5-turbo"):
            # gpt-3.5-turbo currently points to gpt-3.5-turbo-0613. Will point to
            # gpt-3.5-turbo-1106 starting Dec 11, 2023
            cost = (num_tokens * 0.001) / 1000
        elif self._model == "gpt-4":
            cost = (num_tokens * 0.03) / 1000
        elif self._model == "gpt-4-32k":
            cost = (num_tokens * 0.06) / 1000
        else:
            raise ValueError(f"Model {self._model} cost unknown.")

        return cost

    def calculate_cost_output(self, num_tokens: int) -> float:
        """Return the cost of outputting `num_tokens` from the model. This should be
        updated regularly according to https://openai.com/pricing. As new models are
        released, their pricing information should be added to this function.

        Last update: Nov 29, 2023.
        """
        if (self._model == "gpt-3.5-turbo-1106") | (self._model == "gpt-3.5-turbo"):
            # gpt-3.5-turbo currently points to gpt-3.5-turbo-0613. Will point to
            # gpt-3.5-turbo-1106 starting Dec 11, 2023
            cost = (num_tokens * 0.002) / 1000
        elif self._model == "gpt-4":
            cost = (num_tokens * 0.06) / 1000
        elif self._model == "gpt-4-32k":
            cost = (num_tokens * 0.12) / 1000
        else:
            raise ValueError(f"Model {self._model} cost unknown.")

        return cost
