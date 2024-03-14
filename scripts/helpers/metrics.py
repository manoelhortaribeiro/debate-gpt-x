import numpy as np
import pandas as pd
import scipy
from sklearn.metrics import confusion_matrix


def get_bootstrap(
    df,
    true_column: str = "more_convincing_arguments",
    predict_column: str = "response",
    num_bootstraps: int = 1000,
    sample_size: int = 100,
):
    cm = confusion_matrix(
        df[true_column], df[predict_column], labels=["Pro", "Con", "Tie", "other"]
    )
    recalls = (cm.diagonal() / cm.sum(axis=1) * 100).round(2)
    precisions = (cm.diagonal() / cm.sum(axis=0) * 100).round(2)
    accuracy = 100 * cm.diagonal().sum() / cm.sum()

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
