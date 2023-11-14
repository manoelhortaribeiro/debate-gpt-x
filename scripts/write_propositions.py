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


def get_debate_text(debate_id: int, rounds_df: pd.DataFrame) -> str:
    """Returns the text of debate with id 'debate_id' which is stored in multiple rows
    in 'rounds_df'."""
    # retrieve all the rounds of the debate
    debate_rounds = rounds_df[rounds_df.debate_id == debate_id]
    debate_text = ""  # initialize debate text
    for _, round in debate_rounds.iterrows():
        debate_text += round.side + ": " + round.text + "\n\n"

    return debate_text


def update_ids(debate_ids: list[int], path_to_file: str) -> list[int]:
    """Return a list containing all debate ids in 'debate_ids' that are not already in
    'path_to_file'.

    'path_to_file' should be the JSON file where the debate propositions are stored in
    the format {"debate_id": "insert debate_id", "proposition": "insert proposition"}.
    """
    propositions_df = pd.read_json(path_to_file)

    # create list of debate ids already in the file
    debate_ids_file = list(propositions_df.debate_id)
    debate_ids_new = [
        debate_id for debate_id in debate_ids if debate_id not in debate_ids_file
    ]

    return debate_ids_new


def write_proposition(
    debate_id: int,
    debates_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    path_to_file: str,
):
    debate = get_debate_text(debate_id, rounds_df)
    print(f"DEBATE {debate_id}")
    print(
        f"Debate Title: {debates_df[debates_df.debate_id == debate_id].iloc[0].at['title']}\n"  # noqa: E501
    )
    print(f"{debate}\n\n\n")
    proposition = input()
    results = [{"debate_id": debate_id, "proposition": proposition}]
    save_results_to_file(results, path_to_file)


def write_propositions(
    debate_ids: list[int],
    debates_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    path_to_file: str,
) -> None:
    """Write the proposition corresponding to each debate id in the 'debate_ids' list in
    'path_to_file'.

    For each debate in the list of 'debate_ids', the debate itself is printed for the
    user from the rounds contained in 'rounds_df'. Then, the user input is the
    proposition
    that is written to 'path_to_file'."""
    for debate_id in tqdm.tqdm(debate_ids):
        write_proposition(debate_id, debates_df, rounds_df, path_to_file)


def main():
    debates_df = pd.read_json("data/filtered_data/debates_filtered_df.json")
    rounds_df = pd.read_json("data/processed_data/rounds_df.json")
    debate_ids = list(debates_df[debates_df.category == "Politics"].debate_id)
    path_to_file = "data/raw_data/propositions.json"
    debate_ids = update_ids(debate_ids, path_to_file)
    write_propositions(debate_ids, debates_df, rounds_df, path_to_file)


if __name__ == "__main__":
    main()
