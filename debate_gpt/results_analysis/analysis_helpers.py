import warnings

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import t
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")


def get_train_test(
    train_index,
    test_index,
    debate_ids,
    df_dummies,
    output: str = "agreed_before",
):

    train_set = debate_ids[train_index]
    test_set = debate_ids[test_index]
    features = [
        col for col in df_dummies if col not in ["debate_id", "voter_id", output]
    ]
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


def calculate_fleiss_kappa(
    df: pd.DataFrame, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Calculate Fleiss' kappa and the (1-`alpha`)% confidence interval for the data in
    `df`.

    Each row in `df` should represent one sample to be labeled. There should at least
    three columns representing the raters (or models). Row i, column j of the dataframe
    should contain the label given by rater j on sample i. The Fleiss' kappa is
    calculated following the following resource:
    https://audhiaprilliant.medium.com/cohens-kappa-and-fleiss-kappa-how-to-measure-the-agreement-between-raters-9ec12edef121.

    :returns: a float representing the Fleiss' kappa value
              a float representing the lower bound of Fleiss' kappa value for the CI
              a float representing the upper bound of Fleiss' kappa value for the CI
    """

    if len(df.columns) < 3:
        raise ValueError("Dataframe should contain at least 3 columns.")

    num_raters = len(df.columns)
    num_samples = len(df)

    # create new dataframe called `value_counts` where each row contains a sample and
    # each column represents one label. Entry i,j contains number of raters labeling
    # sample i with label j.
    value_counts = df.apply(pd.Series.value_counts, axis=1).fillna(0)
    sample_proportions = ((value_counts**2).sum(axis=1) - num_raters) / (
        num_raters * (num_raters - 1)
    )
    sample_proportions_mean = sample_proportions.mean()

    q_values = value_counts.sum(axis=0) / (num_raters * num_samples)
    proportion_of_errors = (q_values**2).sum()

    fleiss_kappa = (sample_proportions_mean - proportion_of_errors) / (
        1 - proportion_of_errors
    )

    # calculate confidence interval
    variance_kappa = (
        2
        * (
            proportion_of_errors
            - ((2 * num_raters - 3) * (proportion_of_errors**2))
            + (2 * (num_raters - 2) * (q_values**3).sum())
        )
        / (
            (num_raters * num_samples * (num_raters - 1))
            * ((1 - proportion_of_errors) ** 2)
        )
    )
    z_critical = scipy.stats.norm.ppf(1 - alpha / 2)
    fleiss_kappa_lb = fleiss_kappa - z_critical * np.sqrt(variance_kappa)
    fleiss_kappa_ub = fleiss_kappa + z_critical * np.sqrt(variance_kappa)

    return fleiss_kappa, fleiss_kappa_lb, fleiss_kappa_ub


def calculate_cohens_kappa(
    df: pd.DataFrame, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Calculate Cohen's kappa and the 1-`alpha`% confidence interval for the data in
    `df`.

    Each row in `df` should represent one sample to be labeled. There should be two
    and only two columns representing the two raters (or models). Row i, column j of the
    dataframe should contain the label given by rater j on sample i. The Cohen's kappa
    is calculated following the following resource:
    https://audhiaprilliant.medium.com/cohens-kappa-and-fleiss-kappa-how-to-measure-the-agreement-between-raters-9ec12edef121.

    :returns: a float representing the Cohen's kappa value
              a float representing the lower bound of Cohen's kappa value for the CI
              a float representing the upper bound of Cohen's kappa value for the CI
    """

    if len(df.columns) != 2:
        raise ValueError(
            "Dataframe should only contain two columns corresponding to two raters."
        )

    num_samples = len(df)

    possible_labels = pd.unique(df.values.ravel("K"))
    num_labels = len(possible_labels)

    observed_proportions = np.zeros((num_labels, num_labels))
    expected_probabilities = np.zeros((num_labels, num_labels))
    for i in range(num_labels):
        for j in range(num_labels):
            observed_proportions[i, j] = len(
                df[
                    (df[df.columns[0]] == possible_labels[i])
                    & (df[df.columns[1]] == possible_labels[j])
                ]
            )
            expected_probabilities[i, j] = (
                (len(df[df[df.columns[0]] == possible_labels[i]]))
            ) * (len(df[df[df.columns[1]] == possible_labels[j]]))

    observed_proportions = observed_proportions / num_samples
    expected_probabilities = expected_probabilities / num_samples**2

    # calculate cohens kappa
    observed_agreement = observed_proportions.diagonal().sum()
    expected_agreement = np.sum(
        expected_probabilities.sum(axis=1) * expected_probabilities.sum(axis=0)
    )

    cohens_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    # calculate confidence interval
    z_critical = scipy.stats.norm.ppf(1 - alpha / 2)

    standard_error = np.sqrt(
        (observed_agreement * (1 - observed_agreement))
        / (((1 - expected_agreement) ** 2) * num_samples)
    )

    cohens_kappa_lb = cohens_kappa - z_critical * standard_error
    cohens_lappa_ub = cohens_kappa + z_critical * standard_error

    return cohens_kappa, cohens_kappa_lb, cohens_lappa_ub


def get_bootstrap(
    df,
    true_column: str = "ground_truth",
    predict_column: str = "processed_gpt_response",
    num_bootstraps: int = 1000,
):
    sample_size = len(df)
    cm = confusion_matrix(
        df[true_column], df[predict_column], labels=["Pro", "Con", "Tie"]
    )
    recalls = (cm.diagonal() / cm.sum(axis=1) * 100).round(2)

    precisions = (cm.diagonal() / cm.sum(axis=0) * 100).round(2)
    accuracy = 100 * (cm.diagonal().sum() / cm.sum())

    statistics = []
    for _ in range(num_bootstraps):
        sample = df.sample(sample_size, replace=True)
        cm = confusion_matrix(sample[true_column], sample[predict_column])
        acc = cm.diagonal().sum() / cm.sum() * 100
        statistics.append(acc)

    return (
        round(accuracy, 2),
        recalls,
        precisions,
        (round(sorted(statistics)[25], 2), round(sorted(statistics)[975], 2)),
    )
