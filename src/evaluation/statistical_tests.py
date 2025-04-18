import numpy as np
from scipy.stats import ttest_rel, wilcoxon, norm


def paired_t_test(err1: np.ndarray, err2: np.ndarray):
    """
    Paired Student's t-test on two sets of errors.
    Returns: (t_statistic, p_value).
    """
    stat, pval = ttest_rel(err1, err2)
    return stat, pval


def wilcoxon_test(err1: np.ndarray, err2: np.ndarray):
    """
    Wilcoxon signed-rank test on two sets of errors.
    Returns: (statistic, p_value).
    """
    stat, pval = wilcoxon(err1, err2)
    return stat, pval


def diebold_mariano(err1: np.ndarray, err2: np.ndarray, h: int = 1, power: float = 1.0):
    """
    Diebold-Mariano test for predictive accuracy.

    err1, err2: arrays of forecast errors for two models
    h: forecast horizon
    power: loss function power (1 for absolute error, 2 for squared error)

    Returns: (dm_statistic, p_value)
    """
    # loss differential
    d = np.abs(err1)**power - np.abs(err2)**power
    mean_d = np.mean(d)
    n = len(d)
    # compute lagged autocovariance term
    def autocov(d, lag):
        return np.sum((d[lag:] - mean_d) * (d[:n-lag] - mean_d)) / n
    gamma0 = autocov(d, 0)
    gamma_sum = sum(autocov(d, lag) for lag in range(1, h))
    var_d = gamma0 + 2 * gamma_sum
    dm_stat = mean_d / np.sqrt(var_d / n)
    # two-sided p-value under normal approx
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return dm_stat, p_value
