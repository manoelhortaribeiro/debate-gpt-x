import argparse
import json
import os
import sys
import warnings

import pandas as pd

sys.path.append(".")


from debate_gpt.prompt_classes.debate_demographics import (  # noqa: E402, E501
    DebateDemographics,
)
from debate_gpt.prompt_classes.proposition_voter import (  # noqa: E402, E501
    PropositionVoter,
)
from debate_gpt.prompt_classes.who_won import WhoWon  # noqa: E402

warnings.filterwarnings("ignore")


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

    parser.add_argument(
        "--debates",
        type=str,
        default="full",
        help="Should be full, abortion, gay, capital, or issues.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="q2",
        help="Should be q1, q2, q2_prompts, or q3.",
    )

    parser.add_argument("--reasoning", type=str, default="false")
    parser.add_argument("--big_issues", type=str, default="false")
    parser.add_argument("--binary", type=str, default="false")
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
    binary: str,
    reasoning: str,
    big_issues: str,
    propositions_df: pd.DataFrame,
    debates_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    users_df: pd.DataFrame,
    source: str,
    model: str,
    debate_ids: list[int],
    path_to_file: str,
):
    if binary == "true":
        reason_config = task_config["PropositionVoterBinary"]
    elif reasoning == "true":
        reason_config = task_config["PropositionVoterReasoning"]
    else:
        reason_config = task_config["PropositionVoter"]

    if big_issues == "true":
        big_issues_config = task_config["big_issue_columns"]
    else:
        big_issues_config = None

    task = PropositionVoter(
        task_config=reason_config,
        propositions_df=propositions_df,
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

    with open("config/task_configs.json") as f:
        task_config = json.load(f)

    users_df = pd.read_json(task_config["path_to_users"])
    votes_df = pd.read_json(task_config["path_to_votes"])
    if args.binary == "true":
        votes_df = votes_df[votes_df.agreed_before != "Tie"]
    debates_df = pd.read_json(task_config["path_to_debates"])
    rounds_df = pd.read_json(task_config["path_to_rounds"])

    if args.debates == "full":
        propositions_df = pd.read_json(task_config["path_to_propositions"])
    elif args.debates == "abortion":
        propositions_df = pd.read_json(task_config["path_to_abortion_props"])
    elif args.debates == "gay":
        propositions_df = pd.read_json(task_config["path_to_gay_props"])
    elif args.debates == "capital":
        propositions_df = pd.read_json(task_config["path_to_capital_props"])
    elif args.debates == "issues":
        propositions_df = pd.read_json(task_config["path_to_issues_props"])

    debate_ids = list(propositions_df.debate_id.unique())

    # TODO: check also if voter id for each debate has been used
    if args.question != "q2_prompts":
        if os.path.isfile(args.path_to_file):
            with open(args.path_to_file) as f:
                debate_ids_old = list(pd.read_json(f).debate_id.unique())
                debate_ids = [
                    debate for debate in debate_ids if debate not in debate_ids_old
                ]

    # Q1: Can LLMs judge the quality of arguments (compared to humans)?

    if args.question == "q1":
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

    if args.question == "q2":
        proposition_voter(
            task_config=task_config,
            binary=args.binary,
            reasoning=args.reasoning,
            big_issues=args.big_issues,
            propositions_df=propositions_df,
            debates_df=debates_df,
            rounds_df=rounds_df,
            votes_df=votes_df,
            users_df=users_df,
            source=args.source,
            model=args.model,
            debate_ids=debate_ids,
            path_to_file=args.path_to_file,
        )

    if args.question == "q2_prompts":

        for reasoning in ["true", "false"]:
            for big_issues in ["true", "false"]:
                path_to_file = args.path_to_file.split(".json")[0]
                if reasoning == "true":
                    path_to_file += "-r"
                if big_issues == "true":
                    path_to_file += "-bi"

                path_to_file += ".json"
                if os.path.isfile(args.path_to_file):
                    with open(args.path_to_file) as f:
                        debate_ids_old = list(pd.read_json(f).debate_id.unique())
                        debate_ids_new = [
                            debate
                            for debate in debate_ids
                            if debate not in debate_ids_old
                        ]
                proposition_voter(
                    task_config=task_config,
                    binary=args.binary,
                    reasoning=reasoning,
                    big_issues=big_issues,
                    propositions_df=propositions_df,
                    debates_df=debates_df,
                    rounds_df=rounds_df,
                    votes_df=votes_df,
                    users_df=users_df,
                    source=args.source,
                    model=args.model,
                    debate_ids=debate_ids_new,
                    path_to_file=path_to_file,
                )

    # Q3: Do demographics and beliefs improve LLM judging quality?
    if args.question == "q3":
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
