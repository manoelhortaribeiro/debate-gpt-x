import pandas as pd


def create_comments_df(debates_df: pd.DataFrame) -> pd.DataFrame:
    """Return a pandas dataframe containing all information pertinent to comments on
    each debate in `debates_df`.

    `debates_df` should contain at least the following columns: comments, debate_id,
    pro_user_id, and con_user_id.

    :returns: a pandas dataframe
    """
    comments = []
    for _, row in debates_df[
        ["comments", "debate_id", "pro_user_id", "con_user_id"]
    ].iterrows():
        # extract information relevant to the debate
        debate_id = row["debate_id"]
        participant_1 = row["pro_user_id"]
        participant_2 = row["con_user_id"]

        for _, comment in enumerate(row["comments"]):
            comments.append(
                {
                    "debate_id": debate_id,
                    "pro_user_id": participant_1,
                    "con_user_id": participant_2,
                    "commenter_id": comment["user_name"],
                    "comment": comment["comment_text"],
                }
            )

    # create the dataframe
    comments_df = pd.json_normalize(comments)

    # drop all comments that are from the participants of the debate themselves
    comments_df = comments_df[
        (comments_df.pro_user_id != comments_df.commenter_id)
        & (comments_df.con_user_id != comments_df.commenter_id)
    ]
    return comments_df
