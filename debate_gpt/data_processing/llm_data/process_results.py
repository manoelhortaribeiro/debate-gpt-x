import json
import re

import pandas as pd


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


def create_df(files: list[str]) -> pd.DataFrame:
    dfs = []
    for file in files:
        filename = file.split("/")[-1].split(".json")[0]
        model = filename.split("-q")[0]
        df = pd.read_json(file)
        if "r" in file:
            df["gpt_response"] = df.gpt_response.apply(
                lambda x: x.title().split("Answer: ")[-1]
            )
        df["model"] = model
        dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


def extract_answer(x):

    response = x.gpt_response.lower()
    if response == "pro":
        return "Pro"
    if response == "con":
        return "Con"
    if response == "tie":
        return "Tie"

    regexp_pro = r"\bpro\b"
    regexp_con = r"\bcon\b"
    regexp_tie = r"\btie\b"

    if re.search(regexp_pro, response):
        if not re.search(regexp_con, response):
            if not re.search(regexp_tie, response):
                return "Pro"

    elif re.search(regexp_con, response):
        if not re.search(regexp_pro, response):
            if not re.search(regexp_tie, response):
                return "Con"

    elif re.search(regexp_tie, response):
        if not re.search(regexp_con, response):
            if not re.search(regexp_pro, response):
                return "Tie"

    if (
        (re.search(regexp_pro, response) is None)
        & (re.search(regexp_con, response) is None)
        & (re.search(regexp_tie, response) is None)
    ):
        return "Other"

    if ("agree with the ") in response:
        stance = response.split("agree with the ")[1].split("side")[0]
        if "pro" in stance:
            return "Pro"
        elif "con" in stance:
            return "Con"
        elif "tie" in stance:
            return "Tie"

    if "my answer is " in response:
        stance = response.split("my answer is ")[1]
        if "pro" in stance:
            return "Pro"
        elif "con" in stance:
            return "Con"
        elif "tie" in stance:
            return "Tie"

    print("\n\n", x.debate_id, "\n\n", response)
    replacement = input()
    print("got")
    if replacement in ["Pro", "Con", "Tie"]:
        return replacement
    else:
        return "Other"


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


def get_overlap_sets(dfs):
    dfs = [df.groupby(["model"]).debate_id.unique() for df in dfs]
    sets = [el for ls in [[set(sds) for sds in df] for df in dfs] for el in ls]
    debate_set = sets[0]
    for sds in sets:
        debate_set = debate_set.intersection(sds)
    return list(map(int, debate_set))
