import argparse
import glob
import json
import sys

import pandas as pd

sys.path.append(".")
from debate_gpt.data_processing.llm_data.process_results import (  # noqa: E402
    create_df,
    extract_answer,
    get_overlap_sets,
    majority_vote,
    process_crowdsourcing_data,
    to_stance,
)


def prepare_crowd(files: list[str]):
    df = pd.concat([pd.read_csv(file) for file in files]).reset_index(drop=True)
    df = process_crowdsourcing_data(df)
    df = df.groupby(["debate_id", "voter_id"]).sample(1).reset_index(drop=True)
    df["model"] = "MTurk"

    return df


def prepare_ground_truth(
    votes_df: pd.DataFrame, question: str, truth_column: str
) -> pd.DataFrame:

    columns = ["debate_id", "voter_id", truth_column]
    truth_df = votes_df[columns]

    if question == "q1":
        truth_df = (
            truth_df.groupby("debate_id")
            .apply(lambda x: majority_vote(x, truth_column))
            .reset_index(name=truth_column)
        )

    truth_df = truth_df.rename(columns={truth_column: "ground_truth"})
    return truth_df


def prepare_dataframe(
    files: list[str],
    votes_df: pd.DataFrame,
    crowd_df: pd.DataFrame,
    question: str,
    truth_column: str,
    issues: bool = False,
) -> pd.DataFrame:

    ground_truth_df = prepare_ground_truth(votes_df, question, truth_column)

    df = create_df(files, issues)

    df["processed_gpt_response"] = df.apply(lambda x: extract_answer(x), axis=1)

    id_cols = ["debate_id", "voter_id"]
    if question == "q1":
        id_cols = ["debate_id"]

    crowd_df = crowd_df[["debate_id", "voter_id", "model", question]].rename(
        columns={question: "processed_gpt_response"}
    )

    debate_ids = list(df.debate_id.unique())
    df = pd.concat([df, crowd_df[crowd_df.debate_id.isin(debate_ids)]])
    df = df.merge(ground_truth_df, on=id_cols)

    columns = (
        ["model"] + id_cols + ["gpt_response", "processed_gpt_response", "ground_truth"]
    )
    if issues:
        columns += ["big_issues", "reasoning"]
    df = df[columns]

    return df


def prepare_datasets(q1_files, q2_files, q3_files, issues_files):

    q1_df = create_df(q1_files)
    q2_df = create_df(q2_files)
    q3_df = create_df(q3_files)

    datasets = {}
    for debate_length in list(q1_df.debate_length.unique()):
        q1_temp = q1_df[q1_df.debate_length == debate_length]
        q2_temp = q2_df[q2_df.debate_length == debate_length]
        q3_temp = q3_df[q3_df.debate_length == debate_length]
        debate_set = get_overlap_sets([q1_temp, q2_temp, q3_temp])
        datasets[debate_length.title()] = debate_set

    datasets["All"] = get_overlap_sets([q1_df, q2_df, q3_df])

    for file in issues_files:
        props = pd.read_json(file)
        debate_ids = list(props.debate_id.unique())
        name = file.split("/")[-1].split("_props")[0]
        if "_" in name:
            name = name.replace("_", " ")
        name = name.title()
        datasets[name] = list(map(int, debate_ids))

    return datasets


def prepare_regression_dataframes(votes_df, users_df, debates_df, stance_df):

    debate_ids = list(stance_df.debate_id.unique())
    df = votes_df[votes_df.debate_id.isin(debate_ids)][
        ["debate_id", "voter_id", "agreed_before"]
    ]

    # merge df with user demographics
    users_df = users_df.reset_index().rename(columns={"index": "voter_id"})
    df = df.merge(users_df, on="voter_id")

    # merge with debates df to add start date for age calculation
    debates_df["start_date"] = pd.to_datetime(debates_df["start_date"])
    df = df.merge(debates_df[["debate_id", "start_date"]], on="debate_id")

    # age calculation
    df["birthday"] = pd.to_datetime(df.birthday)
    df["age"] = (df.start_date - df.birthday) / pd.Timedelta("365 days")
    df["age"] = (df.age.max() - df.age) / (df.age.max() - df.age.min())
    df["age"] = df.age.fillna(df.age.mean())

    df = df.merge(stance_df, on="debate_id")

    df["agreed_before"] = df.apply(lambda x: to_stance(x, "agreed_before"), axis=1)
    df["stance"] = df.apply(lambda x: to_stance(x, "stance"), axis=1)
    df["agreed_before"] = df.agreed_before * df.stance

    df = df.drop(columns=["start_date", "proposition", "stance", "birthday"])

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_q1", default="false")
    parser.add_argument("--prepare_q2", default="false")
    parser.add_argument("--prepare_q3", default="false")
    parser.add_argument("--prepare_binary", default="false")
    parser.add_argument("--prepare_issues", default="false")
    parser.add_argument("--prepare_datasets", default="false")
    parser.add_argument("--prepare_regression_df", default="false")
    parser.add_argument("--path_to_files", default="data/tidy/llm_outputs")

    args = parser.parse_args()

    votes_df = pd.read_json("data/processing/processed_data/votes_df.json")
    crowd_df = prepare_crowd(glob.glob("data/processing/crowdsourcing/output/*.csv"))

    if args.prepare_q1 == "true":
        files = glob.glob("data/processing/llm_outputs/q1/*.json")
        df = prepare_dataframe(
            files, votes_df, crowd_df, "q1", "more_convincing_arguments"
        )
        df.to_json(args.path_to_files + "/q1.json")

    if args.prepare_q2 == "true":
        files = glob.glob("data/processing/llm_outputs/q2/*.json")
        df = prepare_dataframe(files, votes_df, crowd_df, "q2", "agreed_before")
        df.to_json(args.path_to_files + "/q2.json")

    if args.prepare_q3 == "true":
        files = glob.glob("data/processing/llm_outputs/q3/*.json")
        df = prepare_dataframe(files, votes_df, crowd_df, "q3", "agreed_after")
        df.to_json(args.path_to_files + "/q3.json")

    if args.prepare_binary == "true":
        files = glob.glob("data/processing/llm_outputs/binary/*.json")
        df = prepare_dataframe(files, votes_df, crowd_df, "q2", "agreed_before")
        df.to_json(args.path_to_files + "/binary.json")

    if args.prepare_issues == "true":
        files = glob.glob("data/processing/llm_outputs/issues/*.json")
        df = prepare_dataframe(files, votes_df, crowd_df, "q2", "agreed_before", True)
        df.to_json(args.path_to_files + "/issues.json")

    if args.prepare_datasets == "true":
        datasets = prepare_datasets(
            glob.glob("data/processing/llm_outputs/q1/*.json"),
            glob.glob("data/processing/llm_outputs/q2/*.json"),
            glob.glob("data/processing/llm_outputs/q3/*.json"),
            glob.glob("data/processing/propositions/*_props.json"),
        )

        with open("data/tidy/datasets.json", "w") as f:
            json.dump(datasets, f)

    if args.prepare_regression_df == "true":
        users_df = pd.read_json("data/processing/processed_data/users_df.json")
        debates_df = pd.read_json("data/processing/processed_data/debates_df.json")
        stances = glob.glob("data/processing/propositions/*_props*")
        for stance in stances:
            name = stance.split("/")[-1].split("_props")[0]
            stance_df = pd.read_json(stance)
            df = prepare_regression_dataframes(
                votes_df, users_df, debates_df, stance_df
            )
            df.to_json(args.path_to_files + "/" + name + ".json")


if __name__ == "__main__":
    main()
