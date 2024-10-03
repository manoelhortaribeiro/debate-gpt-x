# Data and Code for: "Can Language Models Recognize Convincing Arguments?"

## Reproduce Results

In order to reproduce our results, only the data in `tidy.zip` on Zenodo is needed (see [Data](#data)).
All the analysis is done directly in the notebook [`analyses.ipynb`](analyses/analyses.ipynb).
Some helper functions are required to run this notebook and are all contained in our debate-gpt package in the file [`analysis_helpers.py`](debate_gpt/results_analysis/analysis_helpers.py).

## Data

Data for this work is available through [Zenodo](https://zenodo.org/records/13887286).

### Raw Data

The `raw/` folder contains `debates.json,` `users.json,` and `readme.md.`
This data was collected in [A Corpus for Modeling User and Language Effects in Argumentation on Online Debating](https://aclanthology.org/P19-1057) (Durmus & Cardie, ACL 2019) and forms the basis for our work.

### Processing

The `processing/` folder contains all intermediate processing applied to our data before the final tidy results.

- `crowdsourcing/` contains two folders `input/` and `output/` which contain the files used for setting up the crowdsourcing tasks on Amazon Mechanical Turk and the results for these tasks, respectively. Each file in these folders is a batch of data.
- `filtered_data/` contains the files:
  - `debates_filtered_df.json`: contains metadata about each debate but only for a select subset of debates following the criteria:
        1. were categorized under "Politics",
        2. contained at least 300 total tokens (tokens are counted using the _tiktoken_ library with the GPT-3.5-turbo model encoding),
        3. contained at least two complete rounds,
        4. The debater who spoke the most in the debate did not speak more than 25\% more than the other debater,
        5. the debate had at least three votes.
  - `votes_filtered_df.json`: contains how each voter voted in the debates in the previous file.
- `llm_outputs/` contains all the outputs received from prompting each llm for each task.
There is a folder associated with each task performed.
In each folder, the files are named in the structure `{model}-q{question number}{suffix}.json` (e.g. `gpt-3.5-q2-bi.json`)
The suffix is the empty string in folders `q1/`, `q2/`, and `q3/`.
In the `binary/` folder the suffix is `-binary` and in the `issues/` folder the suffix may be either the empty string, `-bi`, `-bi-r` or `-r`, where `bi` represents big issues present in the prompt and `r` represents reasoning present in the LLM output.
- `processed_data/` contains the files:
  - `comments_df.json`: ultimately unused but contains a row per comment left on each debate.
  - `debates_df.json`: each row representing one debate in the raw dataset with its assocaited metadata
  - `rounds_df.json`: each row contains one argument in the a debate in the raw data with its associated round number and debate id.
  - `users_df.json`: each row contains a user on the debate platform with their associated demographic information.
  - `votes_df.json`: each row contains a vote from a user
- `propositions/` contains the files:
  - `abortion_props.json`: the propositions for each debate related to abortion
  - `gay_marriage_props.json`: the propositions for each debate related to gay marriage
  - `capital_punishment_props.json`: the propositions for each debate related to capital punishment
  - `issues_props.json`: the propositions for each debate related to all issues combined
  - `propositions.json`: all manually created propositions

### Tidy

The `tidy/` folder contains the data in its final form that is directly used for analyses.

- `datasets/` contains only the file `datasets.json` which contains the debate ids for each dataset we specify in our work
- `llm_outputs/` contains the files:
  - `q1.json`: all the llm outputs for research question 1
  - `q2.json`: all the llm outputs for research question 2
  - `q2-binary.json`: all the llm outputs for research question 2 with only "Pro" and "Con" as options.
  - `q2-issues.json`: all the llm outputs for research question 2 with the issues datasets including whether the input contained big issues and whether the output contained reasoning
  - `q3.json`: all the llm outputs for research question 3
- `regression_files` contains many files each containing a dataframe stored in json format for use in a traditional regression.
