import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class SaturationTransformation(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        function_curve: str = "hill",
        c=10,
        midpoint=None,
        scale=1000,
        lambda_=0.01,
        beta=0.01,
    ):
        self.function_curve = function_curve
        self.c = c
        self.midpoint = midpoint
        self.scale = scale
        self.lambda_ = lambda_
        self.beta = beta

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        # X = check_array(X)
        if self.midpoint is None:
            self.midpoint = 0.5 * np.max(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        if self.function_curve == "hill":
            return hill_saturation(X, self.c, self.midpoint)
        elif self.function_curve == "log":
            return log_saturation(X, self.beta, self.midpoint)
        elif self.function_curve == "exponential":
            return exponential_saturation(X, self.lambda_)
        else:
            raise ValueError("Function curve must be hill, log or exponential")


def hill_saturation(x, c, midpoint):
    """
    Parameters:
    X: investimento
    max_value: Saturação máxima
        - Este é o ponto em que aumentar o investimento em mídia não trará mais crescimento significativo no KPI. Pode ser determinado com base em conhecimento de domínio ou otimizado.
    c: Constante de Hill (curvatura)
        - Controla a forma da curva de saturação. Um valor de c maior que 1 faz com que a curva atinja a saturação mais rapidamente.
    Midpoint: ponto de meia saturação
        - Este valor representa o gasto em mídia no qual você atinge 50% do efeito máximo (S). Também pode ser ajustado empiricamente ou otimizado.

    Returns:
    array-like: Efeito do gasto em mídia.
    """
    return (x**c) / (x**c + midpoint**c)


def log_saturation(x, beta, midpoint):
    """
    Calcula o efeito de um gasto em mídia usando uma curva logarítmica.

    Parameters:
    x (array-like): Gasto em mídia.
    beta (float): Sensibilidade ao gasto.

    Returns:
    array-like: Efeito do gasto em mídia.
    """
    return 1 / (1 + np.exp(-beta * (x - midpoint)))


def exponential_saturation(x, lambda_):
    """
    Calcula o efeito de um gasto em mídia usando uma curva exponencial.

    Parameters:
    x (array-like): Gasto em mídia.
    lambda_ (float): Taxa de crescimento do efeito.

    Returns:
    array-like: Efeito do gasto em mídia.
    """
    return 1 - np.exp(-lambda_ * x)


# if main
if __name__ == "__main__":
    # plot the adstock and the original curve
    max_value = 2
    bins = 0.2
    spend = pd.Series(np.linspace(0, 5000, 11))
    spend.index = np.linspace(0, 5000, 11)

    # plot transformations and the original curve
    # hill
    midpoint = 2000  # half saturation
    c = 5
    # log
    beta = 0.0009
    # exp
    lambda_ = 0.0008

    hill = SaturationTransformation(
        function_curve="hill", midpoint=midpoint, c=c
    ).fit_transform(spend.values.reshape(-1, 1))
    log = SaturationTransformation(
        function_curve="log", beta=beta, midpoint=midpoint
    ).fit_transform(spend.values.reshape(-1, 1))
    exp = SaturationTransformation(
        function_curve="exponential", lambda_=lambda_
    ).fit_transform(spend.values.reshape(-1, 1))
    plt.figure(figsize=(8, 5))
    # plt.plot(spend, label="Original curve")
    plt.plot(spend, hill, label=f"Hill curve with midpoint = {midpoint} and c = {c}")
    plt.plot(spend, log, label=f"Log curve with beta = {beta}")
    plt.plot(spend, exp, label=f"Exp curve with lambda = {lambda_}")
    plt.legend()
    # zoom y from 0 to 1
    plt.ylim(0, 1.2)
    plt.xlim(0, 6000)
    # plt.xticks(spend.index, [round(i, 1) for i in spend.values])
    plt.title("Efeito da Saturação em diferentes funções")
    plt.xlabel("Investimento")
    plt.ylabel("Retorno")
    plt.show()
