import argparse
import json
import os
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
        "--model",
        default="gpt-3.5-turbo-1106",
        type=str,
        help="The specific model to use.",
    )

    parser.add_argument("--debates", type=str, default="full")

    parser.add_argument("--q1", type=str, default="false")
    parser.add_argument("--q2", type=str, default="false")
    parser.add_argument("--q3", type=str, default="false")

    parser.add_argument("--reasoning", type=str, default="false")
    parser.add_argument("--big_issues", type=str, default="false")

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
    reasoning: str,
    big_issues: str,
    debates_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    users_df: pd.DataFrame,
    source: str,
    model: str,
    debate_ids: list[int],
    path_to_file: str,
):
    if reasoning == "true":
        reason_config = task_config["PropositionVoterReasoning"]
    else:
        reason_config = task_config["PropositionVoterRole"]

    if big_issues == "true":
        big_issues_config = task_config["big_issue_columns"]
    else:
        big_issues_config = None

    task = PropositionVoter(
        task_config=reason_config,
        debates_df=debates_df,
        rounds_df=rounds_df,
        votes_df=votes_df,
        users_df=users_df,
        big_issue_columns=big_issues_config,
        demographic_columns=task_config["demographic_columns"],
        demographic_map=task_config["demographics_map"],
        max_gpt_response_tokens=500,
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
        source=source,
        model=model,
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

    if args.debates == "full":
        debate_ids = list(
            debates_df[
                (debates_df.category == "Politics")
                & (debates_df.proposition.str.lower() != "skip")
                & (debates_df.proposition.str.lower() != "drop")
            ].debate_id
        )
    elif args.debates == "abortion":
        debate_ids = list(
            debates_df[
                (debates_df.category == "Politics")
                & (debates_df.proposition.str.lower().str.contains("abortion"))
            ].debate_id
        )

    if os.path.isfile(args.path_to_file):
        with open(args.path_to_file) as f:
            debate_ids_old = list(pd.read_json(f).debate_id.unique())
            debate_ids = [
                debate for debate in debate_ids if debate not in debate_ids_old
            ]

    # Q1: Can LLMs judge the quality of arguments (compared to humans)?

    if args.q1 == "true":
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

    if args.q2 == "true":
        proposition_voter(
            task_config=task_config,
            reasoning=args.reasoning,
            big_issues=args.big_issues,
            debates_df=debates_df,
            rounds_df=rounds_df,
            votes_df=votes_df,
            users_df=users_df,
            source=args.source,
            model=args.model,
            debate_ids=debate_ids,
            path_to_file=args.path_to_file,
        )
    # Q3: Do demographics and beliefs improve LLM judging quality?

    if args.q3 == "true":
        debate_demographics(
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


if __name__ == "__main__":
    main()
