import pandas as pd
import tiktoken


def extract_rounds(debates_df: pd.DataFrame) -> pd.DataFrame:
    """Return a pandas dataframe with each row containing a round in the `debates_df`.

    `debates_df` should contain at least the following columns: debate_id, pro_user_id,
    con_user_id, and rounds.

    :returns: a pandas dataframe
    """
    rounds = []
    for _, row in debates_df[
        ["debate_id", "pro_user_id", "con_user_id", "rounds"]
    ].iterrows():
        # extract details relevant to the debate
        debate_id = row["debate_id"]
        participant_1 = row["pro_user_id"]
        participant_2 = row["con_user_id"]

        order = 0  # keeps track of the order of the conversation
        for i, round in enumerate(row["rounds"]):
            # each round should contain at least one argument and no more than two
            if (len(round) < 1) or (len(round) > 2):
                continue

            order += 1
            rounds.append(
                {
                    "debate_id": debate_id,
                    "round": i,
                    "order": order,
                    "user_id": [
                        participant_1 if round[0]["side"] == "Pro" else participant_2
                    ][0],
                    "side": round[0]["side"],
                    "text": round[0]["text"].replace("\n", "").replace("\r", ""),
                }
            )

            if len(round) == 2:
                order += 1
                rounds.append(
                    {
                        "debate_id": debate_id,
                        "round": i,
                        "order": order,
                        "user_id": [
                            participant_1
                            if round[1]["side"] == "Pro"
                            else participant_2
                        ][0],
                        "side": round[1]["side"],
                        "text": round[1]["text"].replace("\n", "").replace("\r", ""),
                    }
                )

    return pd.json_normalize(rounds)


def add_token_count(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """Add a column called token_count to the `rounds_df` dataframe which contains the
    number of tokens in the text of each round.
    """
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    rounds_df["token_count"] = rounds_df.text.apply(
        lambda x: len(encoding.encode(x)) if x is not None else 0
    )
    return rounds_df


def create_rounds_df(debates_df: pd.DataFrame) -> pd.DataFrame:
    """Return a pandas dataframe with each row containing a round in the `debates_df`
    and add the token count for each round.
    """
    rounds_df = extract_rounds(debates_df)
    rounds_df = add_token_count(rounds_df)
    rounds_df["cum_sum"] = rounds_df.groupby("debate_id").token_count.cumsum()
    return rounds_df
