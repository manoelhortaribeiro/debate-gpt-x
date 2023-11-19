import numpy as np
import pandas as pd
import scipy.stats


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
