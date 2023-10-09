import pandas as pd


def extract_votes(debates_df: pd.DataFrame) -> pd.DataFrame:
    """Return a pandas dataframe with each row containing a vote for a debate in the
    `debates_df`.

    `debates_df` should contain at least the following columns: debate_id, pro_user_id,
    con_user_id, and votes.

    :returns: a pandas dataframe
    """
    votes = []
    for _, row in debates_df[
        ["debate_id", "pro_user_id", "con_user_id", "votes"]
    ].iterrows():
        # extract details relevant to the debate
        debate_id = row["debate_id"]
        participant_1 = row["pro_user_id"]
        participant_2 = row["con_user_id"]

        for vote in row["votes"]:
            user_name = vote["user_name"]
            participants_list = list(vote["votes_map"].keys())
            if len(participants_list) != 3:
                continue

            pro_vote = vote["votes_map"][participant_1]
            con_vote = vote["votes_map"][participant_2]

            votes_map = {
                "debate_id": debate_id,
                "pro_user_id": participant_1,
                "con_user_id": participant_2,
                "voter_id": user_name,
                "pro_vote": pro_vote,
                "con_vote": con_vote,
            }
            votes.append(votes_map)

    votes_df = pd.json_normalize(votes)
    votes_df.columns = (
        votes_df.columns.str.replace(" ", "_").str.lower().str.replace(".", "_")
    )
    return votes_df


def check_votes(row: pd.Series) -> bool:
    """Return true if vote in `row` is valid otherwise return false.

    A vote is valid if for each category of votes (e.g. Who had better conduct), the
    voter only picked one of the two participants or a tie."""
    if (
        row["pro_vote_agreed_with_before_the_debate"]
        + row["con_vote_agreed_with_before_the_debate"]
        > 1
    ):
        return False
    if (
        row["pro_vote_agreed_with_after_the_debate"]
        + row["con_vote_agreed_with_after_the_debate"]
        > 1
    ):
        return False
    if (
        row["pro_vote_who_had_better_conduct"] + row["con_vote_who_had_better_conduct"]
        > 1
    ):
        return False
    if (
        row["pro_vote_had_better_spelling_and_grammar"]
        + row["con_vote_had_better_spelling_and_grammar"]
        > 1
    ):
        return False
    if (
        row["pro_vote_made_more_convincing_arguments"]
        + row["con_vote_made_more_convincing_arguments"]
        > 1
    ):
        return False
    if (
        row["pro_vote_used_the_most_reliable_sources"]
        + row["con_vote_used_the_most_reliable_sources"]
        > 1
    ):
        return False
    return True


def winner(row: pd.Series, col: str) -> str:
    """Return the id of the user that received the vote in `row` for category `col`."""
    if row["pro_vote_" + col]:
        return row.pro_user_id
    if row["con_vote_" + col]:
        return row.con_user_id
    return "tie"


def preprocess_votes_df(votes_df: pd.DataFrame) -> pd.DataFrame:
    """Process `votes_df` to only have valid votes."""
    assert (~votes_df.apply(lambda x: check_votes(x), axis=1)).sum() == 0
    votes_df["agreed_before"] = votes_df.apply(
        lambda x: winner(x, "agreed_with_before_the_debate"), axis=1
    )
    votes_df["agreed_after"] = votes_df.apply(
        lambda x: winner(x, "agreed_with_after_the_debate"), axis=1
    )
    votes_df["better_conduct"] = votes_df.apply(
        lambda x: winner(x, "who_had_better_conduct"), axis=1
    )
    votes_df["better_spelling_and_grammar"] = votes_df.apply(
        lambda x: winner(x, "had_better_spelling_and_grammar"), axis=1
    )
    votes_df["more_convincing_arguments"] = votes_df.apply(
        lambda x: winner(x, "made_more_convincing_arguments"), axis=1
    )
    votes_df["most_reliable_sources"] = votes_df.apply(
        lambda x: winner(x, "used_the_most_reliable_sources"), axis=1
    )
    votes_df = votes_df[
        [
            col
            for col in votes_df.columns
            if not (col.startswith("pro_vote") or col.startswith("con_vote"))
        ]
    ]
    votes_df["flipped"] = votes_df.apply(
        lambda x: True if (x.agreed_before != x.agreed_after) else False, axis=1
    )
    return votes_df


def create_votes_df(debates_df: pd.DataFrame) -> pd.DataFrame:
    """Create the votes dataframe from `debate_df`."""
    votes_df = extract_votes(debates_df)
    votes_df = preprocess_votes_df(votes_df)
    votes_df = votes_df[
        (votes_df.pro_user_id != votes_df.voter_id)
        & (votes_df.con_user_id != votes_df.voter_id)
    ]
    return votes_df
