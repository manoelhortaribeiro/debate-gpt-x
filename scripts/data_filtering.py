import sys

import pandas as pd

sys.path.append(".")
from debate_gpt.data_filtering.filter_data import (  # noqa: E402
    add_propositions,
    filter_by_rounds,
    filter_by_votes,
    filter_votes_by_users,
)


def main():
    debates_df = pd.read_json("data/processed_data/debates_df.json")
    debates_df = debates_df.set_index("debate_id")

    # Filter debates by rounds
    rounds_df = pd.read_json("data/processed_data/rounds_df.json")
    debates_df = filter_by_rounds(rounds_df, debates_df, percentage=25, min_tokens=300)

    # Filter debates by votes
    votes_df = pd.read_json("data/processed_data/votes_df.json")
    debates_df = filter_by_votes(
        votes_df, debates_df, min_num_votes=3, min_num_flipped_votes=0
    )

    # Filter votes by users
    users_df = pd.read_json("data/processed_data/users_df.json")
    votes_df = filter_votes_by_users(
        users_df=users_df,
        votes_df=votes_df,
        debates_df=debates_df,
        min_num_demographics=5,
    )
    votes_df.to_json("data/filtered_data/votes_filtered_df.json", orient="records")

    debates_df = debates_df.reset_index()
    debates_df.to_json("data/filtered_data/debates_filtered_df.json", orient="records")

    # Create dataframe with propositions
    propositions_df = pd.read_json("data/raw_data/propositions.json")
    propositions_df = propositions_df[
        (propositions_df.proposition != "drop")
        & (propositions_df.proposition != "skip")
    ]
    add_propositions(
        debates_df=debates_df,
        propositions_df=propositions_df,
        path_to_file="data/filtered_data/debates_titles.json",
    )


if __name__ == "__main__":
    main()
