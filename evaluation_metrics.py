import numpy as np
from sklearn.metrics import mean_squared_error


def support_confusion_matrix(true_coefs, predicted_coefs):
    true_ixs = set(np.nonzero(true_coefs.flatten())[0])
    predicted_ixs = set(np.nonzero(predicted_coefs.flatten())[0])
    n_true_positives = len(true_ixs.intersection(predicted_ixs))
    n_false_negatives = len(true_ixs - predicted_ixs)
    n_false_positives = len(predicted_ixs - true_ixs)
    return n_true_positives, n_false_positives, n_false_negatives


def support_true_positivity(true_coefs, predicted_coefs):
    tp, fp, fn = support_confusion_matrix(true_coefs, predicted_coefs)
    return tp / (tp + fp + fn)


def coefficient_normalized_linf(true_coefs, predicted_coefs):
    return np.max(np.abs(true_coefs - predicted_coefs))\
           / np.max(np.abs(true_coefs))


def coefficient_normalized_l2(true_coefs, predicted_coefs):
    return np.linalg.norm(true_coefs - predicted_coefs, 'fro')\
           / np.linalg.norm(true_coefs, 'fro')


def coefficient_normalized_l1(true_coefs, predicted_coefs):
    return np.sum(np.abs(true_coefs - predicted_coefs))\
           / np.sum(np.abs(true_coefs))


def AIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=True, add_correction=True):
    rss = np.sum((x_dot_valid - x_dot_pred) ** 2,
                 axis=0 if keep_dimensionalized else None)
    m = x_dot_valid.shape[0] * (1 if keep_dimensionalized else x_dot_valid.shape[1])
    aic = 2 * k + m * np.log(rss / m)
    if add_correction:
        correction_term = (2 * (k + 1) * (k + 2)) / max(m - k - 2, 1)  # In case k == m
        aic += correction_term
    return aic


def BIC(x_dot_valid, x_dot_pred, k, keep_dimensionalized=False):
    rss = np.sum((x_dot_valid - x_dot_pred) ** 2,
                 axis=0 if keep_dimensionalized else None)
    m = x_dot_valid.shape[0] * (1 if keep_dimensionalized else x_dot_valid.shape[1])
    bic = np.log(m) * k + m * np.log(rss / m)
    return bic


def clean_test_scores(model, system, n=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    r2s = []
    mses = []
    x0s = system.sample_initial_conditions(n)
    for x0 in x0s:
        t, x_test = system.simulate(x0=x0)
        r2s.append(model.score(x_test, t=t[1]-t[0]))
        mses.append(model.score(x_test, t=t[1]-t[0], metric=mean_squared_error))
    return np.array(r2s), np.sqrt(np.array(mses))


def constraint_violation(model, constraint):
    # Assumes constraint RHS == 0
    return constraint @ model.optimizer.coef_.flatten(order='F')


