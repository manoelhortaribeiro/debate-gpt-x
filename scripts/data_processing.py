import sys

sys.path.append(".")

from debate_gpt.data_processing.create_comments_df import (  # noqa: E402
    create_comments_df,
)
from debate_gpt.data_processing.create_debates_df import create_debates_df  # noqa: E402
from debate_gpt.data_processing.create_rounds_df import create_rounds_df  # noqa: E402
from debate_gpt.data_processing.create_users_df import create_users_df  # noqa: E402
from debate_gpt.data_processing.create_votes_df import create_votes_df  # noqa: E402


def main():
    PATH_TO_RAW_USERS_DATA = "data/raw_data/users.json"
    PATH_TO_RAW_DEBATES_DATA = "data/raw_data/debates.json"
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
    user_activity_columns = [
        "number_of_all_debates",
        "number_of_won_debates",
        "number_of_voted_debates",
    ]
    users_df = create_users_df(
        PATH_TO_RAW_USERS_DATA, demographic_columns, user_activity_columns
    )
    debates_df = create_debates_df(PATH_TO_RAW_DEBATES_DATA)
    votes_df = create_votes_df(debates_df)
    rounds_df = create_rounds_df(debates_df)
    comments_df = create_comments_df(debates_df)
    debates_df = debates_df[
        ["debate_id", "pro_user_id", "con_user_id", "title", "category"]
    ]

    users_df.to_json("data/processed_data/users_df.json")
    votes_df.to_json("data/processed_data/votes_df.json")
    comments_df.to_json("data/processed_data/comments_df.json")
    rounds_df.to_json("data/processed_data/rounds_df.json")
    debates_df.to_json("data/processed_data/debates_df.json")


if __name__ == "__main__":
    main()
