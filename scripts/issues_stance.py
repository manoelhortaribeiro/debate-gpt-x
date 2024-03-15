import argparse
import json
import os

import pandas as pd
import tqdm


def save_results_to_file(results: list[dict[str, str]], path_to_file: str) -> None:
    """Save `results` into `path_to_file`.

    The file should be a JSON file. If the file already exists, the results will be
    appended to the current contents of the file. Otherwise, a new file will be
    created.
    """
    if os.path.isfile(path_to_file):
        with open(path_to_file) as f:
            results_old = json.load(f)
            results = results_old + results

    with open(path_to_file, "w") as f:
        json.dump(results, f)


def write_stance(propositions_df: pd.DataFrame, debate_id: int, path_to_file: str):
    proposition = propositions_df[
        propositions_df.debate_id == debate_id
    ].proposition.values[0]
    print(proposition)
    stance = input()
    results = [
        {"debate_id": str(debate_id), "proposition": proposition, "stance": stance}
    ]
    save_results_to_file(results, path_to_file)
    pass


def write_stances(
    propositions_df: pd.DataFrame, debate_ids: list[int], path_to_file: str
):
    for debate_id in tqdm.tqdm(debate_ids):
        write_stance(propositions_df, debate_id, path_to_file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--issue", default="abortion")
    parser.add_argument(
        "--path_to_file", default="data/processed_data/abortion_props.json"
    )
    args = parser.parse_args()

    propositions_df = pd.read_json("data/raw_data/propositions.json")
    ABORTION = list(
        propositions_df[
            (propositions_df.proposition.str.lower().str.contains("abortion"))
        ].debate_id.unique()
    )
    GAY_MARRIAGE = list(
        propositions_df[
            (
                propositions_df.proposition.str.lower().str.contains("same sex")
                | propositions_df.proposition.str.lower().str.contains("gay")
                | propositions_df.proposition.str.lower().str.contains("same-sex")
            )
            & (propositions_df.proposition.str.lower().str.contains("marriage"))
        ].debate_id.unique()
    )

    CAPITAL_PUNISHMENT = list(
        propositions_df[
            (
                propositions_df.proposition.str.lower().str.contains("death penalty")
                | propositions_df.proposition.str.lower().str.contains(
                    "capital punishment"
                )
            )
        ].debate_id.unique()
    )

    if args.issue == "abortion":
        debate_ids = ABORTION
    elif args.issue == "gay_marriage":
        debate_ids = GAY_MARRIAGE
    elif args.issue == "capital_punishment":
        debate_ids = CAPITAL_PUNISHMENT
    else:
        raise ValueError("Issue must be abortion, gay marriage, or capital punishment.")

    write_stances(propositions_df, debate_ids, args.path_to_file)


if __name__ == "__main__":
    main()
