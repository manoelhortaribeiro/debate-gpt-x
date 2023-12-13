import argparse
import json
import sys

import pandas as pd

sys.path.append(".")

from debate_gpt.prompt_classes.openai_prompts.debate_demographics import (  # noqa: E402, E501
    DebateDemographics,
)
from debate_gpt.prompt_classes.openai_prompts.proposition_voter import (  # noqa: E402
    PropositionVoter,
)
from debate_gpt.prompt_classes.openai_prompts.who_won import WhoWon  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="openai",
        type=str,
        help="Should be either 'openai' or 'opensource'.",
    )

    parser.add_argument(
        "--model", default="gpt-3.5-turbo", type=str, help="The specific model to use."
    )

    parser.add_argument("--path_to_file", type=str)
    args = parser.parse_args()
    return args


def who_won(
    task_config,
    debates_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    users_df: pd.DataFrame,
    source: str,
    model: str,
    debate_ids: list[int],
    path_to_file: str,
):
    task = WhoWon(
        task_config=task_config["WhoWon"],
        debates_df=debates_df,
        rounds_df=rounds_df,
        votes_df=votes_df,
        users_df=users_df,
        source=source,
        model=model,
    )
    task.get_batch_results(
        debate_ids=debate_ids,
        path_to_file=path_to_file,
    )


def proposition_voter(
    task_config,
    debates_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    users_df: pd.DataFrame,
    source: str,
    model: str,
    debate_ids: list[int],
    path_to_file: str,
):
    task = PropositionVoter(
        task_config=task_config["PropositionVoterRole"],
        debates_df=debates_df,
        rounds_df=rounds_df,
        votes_df=votes_df,
        users_df=users_df,
        big_issue_columns=None,
        demographic_columns=task_config["demographic_columns"],
        demographic_map=task_config["demographics_map"],
        max_gpt_response_tokens=5,
        source=source,
        model=model,
    )

    task.get_batch_results(debate_ids, path_to_file)


def debate_demographics(
    task_config,
    debates_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    users_df: pd.DataFrame,
    source: str,
    model: str,
    debate_ids: list[int],
    path_to_file: str,
):
    task = DebateDemographics(
        task_config=task_config["DebateDemographics"],
        debates_df=debates_df,
        rounds_df=rounds_df,
        votes_df=votes_df,
        users_df=users_df,
        demographic_columns=task_config["demographic_columns"],
    )
    task.get_batch_results(
        debate_ids=debate_ids,
        path_to_file=path_to_file,
    )


def main():
    args = parse_args()

    with open("scripts/task_configs.json") as f:
        task_config = json.load(f)

    users_df = pd.read_json(task_config["path_to_users"])
    votes_df = pd.read_json(task_config["path_to_votes"])
    rounds_df = pd.read_json(task_config["path_to_rounds"])
    debates_df = pd.read_json(task_config["path_to_debates_with_title"])

    debate_ids = list(debates_df[debates_df.category == "Politics"].debate_id)

    # Q1: Can LLMs judge the quality of arguments (compared to humans)?

    who_won(
        task_config=task_config,
        debates_df=debates_df,
        rounds_df=rounds_df,
        votes_df=votes_df,
        users_df=users_df,
        source=args.source,
        model=args.model,
        debate_ids=debate_ids,
        path_to_file=args.path_to_file,
    )
    # Q2: Can LLMs judge how a person’s demographics and beliefs affect their stance on
    # a topic?

    # Q3: Do demographics and beliefs improve LLM judging quality?


if __name__ == "__main__":
    main()
