import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment


def clustering_accuracy(true_row_labels, predicted_row_labels):
    """
    The :mod:`coclust.evaluation.external` module provides functions
    to evaluate clustering or co-clustering results with external information
    such as the true labeling of the clusters.
    """

    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    rows, cols = linear_assignment(_make_cost_m(cm))
    total = 0
    for row, column in zip(rows, cols):
        value = cm[row][column]
        total += value

    return (total * 1. / np.sum(cm)), cols


def _make_cost_m(cm):
    s = np.max(cm)
    return - cm + s
