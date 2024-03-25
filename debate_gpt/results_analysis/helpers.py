import numpy as np
import pandas as pd
from scipy.stats import t


def get_train_test(
    train_index,
    test_index,
    debate_ids,
    df_dummies,
    features,
    output: str = "agreed_before",
):

    train_set = debate_ids[train_index]
    test_set = debate_ids[test_index]
    df_train = df_dummies[df_dummies.debate_id.isin(train_set)]
    df_test = df_dummies[df_dummies.debate_id.isin(test_set)]
    X_train = pd.get_dummies(df_train[features])
    X_test = pd.get_dummies(df_test[features])

    y_train = df_train[output]
    y_test = df_test[output]

    return X_train, y_train, X_test, y_test


def get_metrics(scores):
    sample_mean = np.mean(scores)
    sample_std = np.std(scores, ddof=1)  # using ddof=1 for sample standard deviation

    # Step 2: Determine the t-value for a 95% confidence interval
    confidence_level = 0.95
    degrees_of_freedom = len(scores) - 1
    t_value = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Step 3: Calculate confidence interval
    margin_of_error = t_value * (sample_std / np.sqrt(len(scores)))
    confidence_interval = (
        round((sample_mean - margin_of_error) * 100, 2),
        round((sample_mean + margin_of_error) * 100, 2),
    )

    return round(sample_mean * 100, 2), confidence_interval
