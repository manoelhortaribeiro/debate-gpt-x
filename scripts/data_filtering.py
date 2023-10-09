import sys

import pandas as pd

sys.path.append(".")
from debate_gpt.data_filtering.filter_data import (  # noqa: E402
    filter_by_rounds,
    filter_by_votes,
    filter_votes_by_users,
)


def main():
    debates_df = pd.read_json("data/processed_data/debates_df.json")
    debates_df = debates_df.set_index("debate_id")

    # Filter debates by rounds
    rounds_df = pd.read_json("data/processed_data/rounds_df.json")
    debates_df = filter_by_rounds(rounds_df, debates_df)

    # Filter debates by votes
    votes_df = pd.read_json("data/processed_data/votes_df.json")
    debates_df = filter_by_votes(votes_df, debates_df)

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


if __name__ == "__main__":
    main()
