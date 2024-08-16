import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from tabulate import tabulate


def r2_score_adjusted(y_true, y_pred, X):
    r2 = r2_score(y_true, y_pred)

    # Número de observações e variáveis independentes
    n = X.shape[0]
    p = X.shape[1]

    # Calcular R² ajustado
    r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return r2_adjusted


# write wape
def weighted_absolute_percentage_error(y_true, y_pred, minimize=True):
    if minimize:
        return -np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    else:
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


# do a report with both metrics
def report_metrics(y_true, y_pred, minimize=False, only_print=True):
    report = pd.Series()
    report["mae"] = (
        "R$ " + str(round(mean_absolute_error(y_true, y_pred) / 1e3, 1)) + "mil"
    )
    report["mape"] = (
        str(round(mean_absolute_percentage_error(y_true, y_pred) * 100, 1)) + "%"
    )
    report["wape"] = (
        str(
            round(weighted_absolute_percentage_error(y_true, y_pred, minimize) * 100, 1)
        )
        + "%"
    )
    report["r2"] = str(round(r2_score(y_true, y_pred) * 100, 1)) + "%"
    # use tabulate
    print(tabulate(report.to_frame(), headers=["Metric", "Value"]))
    if not only_print:
        return report
