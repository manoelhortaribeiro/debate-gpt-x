import json

import pandas as pd


def get_crowd_answer(x):
    if x["Pro"]:
        return "Pro"
    elif x["Con"]:
        return "Con"
    else:
        return "Tie"


def process_crowdsourcing_data(crowd: pd.DataFrame):
    crowd["q1"] = crowd.apply(
        lambda x: json.loads(x["Answer.taskAnswers"])[0]["q3"], axis=1
    )
    crowd["q2"] = crowd.apply(
        lambda x: json.loads(x["Answer.taskAnswers"])[0]["q1"], axis=1
    )
    crowd["q3"] = crowd.apply(
        lambda x: json.loads(x["Answer.taskAnswers"])[0]["q2"], axis=1
    )
    crowd["q1"] = crowd.q1.apply(lambda x: get_crowd_answer(x))
    crowd["q2"] = crowd.q2.apply(lambda x: get_crowd_answer(x))
    crowd["q3"] = crowd.q3.apply(lambda x: get_crowd_answer(x))

    crowd = crowd[["Input.debate_id", "Input.voter_id", "q1", "q2", "q3"]]
    crowd = crowd.rename(
        columns={"Input.debate_id": "debate_id", "Input.voter_id": "voter_id"}
    )
    return crowd


def majority_vote(row, column):
    num_pro = (row[column] == "Pro").sum()
    num_con = (row[column] == "Con").sum()
    num_tie = (row[column] == "Tie").sum()

    if (num_tie > num_pro) & (num_tie > num_con):
        return "Tie"
    elif num_pro > num_con:
        return "Pro"
    elif num_con > num_pro:
        return "Con"
    elif num_pro == num_con:
        return "Tie"
    else:
        return "else"


def process_gpt_response(df: pd.DataFrame):
    df["correct_form"] = df.gpt_response.isin(["Pro", "Con", "Tie"])
    df["gpt_response"] = df.gpt_response.apply(
        lambda x: "Pro" if ("Pro" in x) & ("Con" not in x) & ("Tie" not in x) else x
    )
    df["gpt_response"] = df.gpt_response.apply(
        lambda x: "Con" if ("Con" in x) & ("Pro" not in x) & ("Tie" not in x) else x
    )
    df["gpt_response"] = df.gpt_response.apply(
        lambda x: "Tie" if ("Tie" in x) & ("Pro" not in x) & ("Con" not in x) else x
    )
    df["gpt_response"] = df.gpt_response.apply(
        lambda x: "other" if x not in ["Pro", "Con", "Tie"] else x
    )
    df["answer_extracted"] = df.gpt_response.isin(["Pro", "Con", "Tie"])

    return df


def get_gpt_response(
    df: pd.DataFrame, column: str = "gpt_response", reasoning=False
) -> pd.DataFrame:
    df = df.reset_index(names="vote_id")
    if reasoning:
        df.rename(columns={"gpt_response": "gpt_answer"}, inplace=True)
        df["gpt_response"] = df.gpt_answer.apply(
            lambda x: x.title().split("Answer: ")[-1]
        )
        df = df.drop(columns="gpt_answer")

    df = process_gpt_response(df)
    return df
