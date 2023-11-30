import json
import sys

import pandas as pd

sys.path.append(".")

from debate_gpt.prompt_classes.openai_prompts.debate_demographics import (  # noqa: E402
    DebateDemographics,
)
from debate_gpt.prompt_classes.openai_prompts.who_won import WhoWon  # noqa: E402


def main():
    with open(".config/task_configs.json") as f:
        task_config = json.load(f)

    users_df = pd.read_json(task_config["path_to_users"])
    votes_df = pd.read_json(task_config["path_to_votes"])
    rounds_df = pd.read_json(task_config["path_to_rounds"])
    debates_df = pd.read_json(task_config["path_to_debates_with_title"])

    debate_ids = list(debates_df[debates_df.category == "Politics"].debate_id)

    # Q1: Can LLMs judge the quality of arguments (compared to humans)?

    task = WhoWon(
        task_config=task_config["WhoWon"],
        debates_df=debates_df,
        rounds_df=rounds_df,
        votes_df=votes_df,
        users_df=users_df,
    )
    task.get_batch_results(
        debate_ids=debate_ids,
        path_to_file="results/q1/whowon2.json",
    )

    # Q2: Can LLMs judge how a person’s demographics and beliefs affect their stance on
    # a topic?

    # Q3: Do demographics and beliefs improve LLM judging quality?

    task = DebateDemographics(
        task_config=task_config["DebateDemographics"],
        debates_df=debates_df,
        rounds_df=rounds_df,
        votes_df=votes_df,
        users_df=users_df,
        demographic_columns=task_config["demographic_columns"],
    )
    # task.get_batch_results(
    #     debate_ids=debate_ids,
    #     path_to_file="results/q3/debate_demographics.json",
    # )


if __name__ == "__main__":
    main()
