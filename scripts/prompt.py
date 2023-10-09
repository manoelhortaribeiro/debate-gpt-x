import sys

import pandas as pd

sys.path.append(".")

from debate_gpt.prompt_classes.task_one_class import (  # noqa: E402
    ContextQuestionConstraintClass,
)


def main():
    demographic_columns = [
        "birthday",
        "education",
        "ethnicity",
        "gender",
        "income",
        "interested",
        "party",
        "political_ideology",
        "relationship",
        "religious_ideology",
    ]
    users_df = pd.read_json("data/processed_data/users_df.json")
    votes_df = pd.read_json("data/filtered_data/votes_filtered_df.json")
    rounds_df = pd.read_json("data/processed_data/rounds_df.json")
    debates_df = pd.read_json("data/filtered_data/debates_filtered_df.json")

    task_one = ContextQuestionConstraintClass(
        demographic_columns=demographic_columns,
        users_df=users_df,
        votes_df=votes_df,
        rounds_df=rounds_df,
        debates_df=debates_df,
    )

    debate_ids = list(
        task_one.debates_df[task_one.debates_df.category == "Politics"].debate_id
    )
    path_to_file = "data/prompt_results/politics_debates.json"
    task_one.get_batch_predictions(debate_ids=debate_ids, path_to_file=path_to_file)


if __name__ == "__main__":
    main()
