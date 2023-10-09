import numpy as np
import pandas as pd


def create_debates_df(path_to_data: str) -> pd.DataFrame:
    """Return a pandas dataframe containing the procesed data in json file at
    `path_to_data`.

    :returns: a pandas dataframe.
    """

    raw_debates_df = pd.read_json(path_to_data, orient="index").reset_index(drop=True)

    # drop all columns that can be recalculated properly or are not relevant
    debates_df = raw_debates_df.copy().drop(
        [
            "url",
            "forfeit_label",
            "forfeit_side",
            "start_date",
            "update_date",
            "voting_style",
            "participant_1_link",
            "participant_1_status",
            "participant_2_link",
            "participant_2_status",
            "debate_status",
            "number_of_comments",
            "number_of_views",
            "number_of_rounds",
            "number_of_votes",
            "participant_1_points",
            "participant_2_points",
        ],
        axis=1,
    )

    assert debates_df.participant_1_position.unique() == np.array(["Pro"])
    assert debates_df.participant_2_position.unique() == np.array(["Con"])

    debates_df = debates_df.drop(
        ["participant_1_position", "participant_2_position"], axis=1
    )
    debates_df = debates_df.rename(
        columns={
            "participant_1_name": "pro_user_id",
            "participant_2_name": "con_user_id",
        }
    )
    debates_df["debate_id"] = debates_df.index
    return debates_df
