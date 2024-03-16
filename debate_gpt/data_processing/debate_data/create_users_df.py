import numpy as np
import pandas as pd


def create_demographics_df(
    df: pd.DataFrame, demographic_columns: list[str]
) -> pd.DataFrame:
    """Returns a dataframe with columns `demographic_columns` of each user in `df`.

    Each column in the resulting dataframe should contain some demographic information
    about the user. If no answer provided, replace value with NaN.
    """
    demographics_df = df.copy()[demographic_columns]
    demographics_df = demographics_df.replace(
        ["- Private -", "Not Saying", "Prefer not to say", "No Answer"], np.NaN
    )
    demographics_df["num_demographics"] = (~demographics_df.isna()).sum(axis=1)
    return demographics_df


def create_big_issues_df(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe containing each users response on big issues from `df`."""
    big_issues_df = pd.json_normalize(df.big_issues_dict).set_index(df.index)
    big_issues_df.columns = big_issues_df.columns.str.lower().str.replace(" ", "_")
    big_issues_df = big_issues_df.replace(["N/S"], np.NaN)
    big_issues_df["num_big_issues"] = (~big_issues_df.isna()).sum(axis=1)
    return big_issues_df


def create_user_activity_df(
    df: pd.DataFrame, user_activity_columns: list[str]
) -> pd.DataFrame:
    """Returns a dataframe with columns `user_activity_columns` of each user in `df`.

    Each column in the resulting dataframe should contain some information
    about the user's activity on the debate platform.
    """
    activity_df = df.copy()[user_activity_columns]
    activity_df["number_participated"] = (
        activity_df.number_of_all_debates + activity_df.number_of_voted_debates
    )
    return activity_df


def create_users_df(
    path_to_data: str, demographic_columns: list[str], activity_columns: list[str]
) -> pd.DataFrame:
    """Return a dataframe with each users response to big issues, `demographic_columns`,
    and `activity_columns` from data stored at `path_to_data`.
    """
    df = pd.read_json(path_to_data, orient="index")
    demographics_df = create_demographics_df(df, demographic_columns)
    big_issues_df = create_big_issues_df(df)
    activity_df = create_user_activity_df(df, activity_columns)
    users_df = pd.concat([demographics_df, activity_df, big_issues_df], axis=1)
    return users_df
