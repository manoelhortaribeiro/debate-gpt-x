import pandas as pd


def filter_by_rounds(
    rounds_df: pd.DataFrame,
    debates_df: pd.DataFrame,
    percentage: float = 25,
    min_tokens: int = 300,
) -> pd.DataFrame:
    """Return `debates_df` filtered by the following rules: only debates with at least
    `min_tokens` total are kept, only debates where the debater who speaks the most does
    not speak more than `percentage` percent more than the other debater are kept, and
    only debates containing at least 2 full rounds (each debater speaks at least twice)
    are kept. All this information can be found in `rounds_df`.
    """
    # get total number of tokens per debater in each debate
    debater_token_counts = (
        rounds_df.groupby(["debate_id", "side"])["token_count"]
        .sum()
        .to_frame()
        .reset_index()
        .pivot(index="debate_id", columns="side", values="token_count")
        .reset_index()
    )

    # ensure a balanced debate
    debater_token_counts_balanced = debater_token_counts[
        debater_token_counts.apply(
            lambda x: ((1 + (percentage / 100)) * min(x.Pro, x.Con))
            >= max(x.Pro, x.Con),
            axis=1,
        )
    ]

    # ensure a minimum number of tokens per debate
    debater_token_counts_full = debater_token_counts_balanced[
        debater_token_counts_balanced.apply(
            lambda x: x.Con + x.Pro >= min_tokens, axis=1
        )
    ]
    debate_ids_tokens = list(debater_token_counts_full.debate_id.unique())

    # ensure two full rounds
    rounds_size = rounds_df.groupby("debate_id").size()
    debate_ids_size = list(rounds_size[rounds_size >= 4].index)

    # get list of debate ids that follow these parameters
    debate_ids = [value for value in debate_ids_tokens if value in debate_ids_size]

    return debates_df[debates_df.index.isin(debate_ids)]


def filter_by_votes(
    votes_df: pd.DataFrame,
    debates_df: pd.DataFrame,
    min_num_votes: int = 3,
    min_num_flipped_votes: int = 0,
) -> pd.DataFrame:
    """Return `debates_df` filtered by qualities of the votes in `votes_df`: only
    debates with at least `min_num_votes` votes are kept and only debates with at least
    `min_num_flipped_votes` are kept (these are the number of voters that changed their
    mind from before to after the debate).
    """
    debate_vote_stats = pd.DataFrame(
        {
            "num_votes": votes_df.groupby(["debate_id"]).size(),
            "num_flipped_votes": votes_df.groupby(["debate_id"]).apply(
                lambda x: (x.flipped).sum()
            ),
            "num_pro_agreed_before": votes_df.groupby(["debate_id"]).apply(
                lambda x: (x.agreed_before == "Pro").sum()
            ),
            "num_con_agreed_before": votes_df.groupby(["debate_id"]).apply(
                lambda x: (x.agreed_before == "Con").sum()
            ),
            "num_tie_agreed_before": votes_df.groupby(["debate_id"]).apply(
                lambda x: (x.agreed_before == "Tie").sum()
            ),
            "num_pro_agreed_after": votes_df.groupby(["debate_id"]).apply(
                lambda x: (x.agreed_after == "Pro").sum()
            ),
            "num_con_agreed_after": votes_df.groupby(["debate_id"]).apply(
                lambda x: (x.agreed_after == "Con").sum()
            ),
            "num_tie_agreed_after": votes_df.groupby(["debate_id"]).apply(
                lambda x: (x.agreed_after == "Tie").sum()
            ),
        }
    )

    debates_df = debates_df.join(debate_vote_stats)
    debates_df = debates_df[debates_df.num_votes >= min_num_votes]
    debates_df = debates_df[debates_df.num_flipped_votes >= min_num_flipped_votes]
    return debates_df


def filter_votes_by_users(
    users_df: pd.DataFrame,
    votes_df: pd.DataFrame,
    debates_df: pd.DataFrame,
    min_num_demographics: int,
) -> pd.DataFrame:
    """Return `votes_df` filtered by debates in `debates_df` and users with at least
    `min_num_demographics` demographic data in `users_df`."""
    votes_df = votes_df.merge(
        users_df[["num_big_issues", "num_demographics", "number_participated"]],
        how="left",
        left_on="voter_id",
        right_index=True,
    )
    votes_df = votes_df[votes_df.debate_id.isin(debates_df.index)]
    votes_df = votes_df[votes_df.num_demographics >= min_num_demographics]

    return votes_df


def add_propositions(
    debates_df: pd.DataFrame, propositions_df: pd.DataFrame, path_to_file: str
) -> None:
    propositions_df.merge(debates_df, on="debate_id").to_json(path_to_file)
