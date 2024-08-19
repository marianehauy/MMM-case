import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class SaturationTransformation(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        function_curve: str = "hill",
        slope_s=5,
        midpoint=0.5,
        lambda_=10,
        beta=1,
        normalize = False
    ):
        self.function_curve = function_curve
        self.slope_s = slope_s
        self.midpoint = midpoint
        self.lambda_ = lambda_
        self.beta = beta
        self.normalize = normalize

    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        return self

    def transform(self, X):
        check_is_fitted(self)
        # X = check_array(X)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        if self.function_curve == "hill":
            return hill_saturation(X, self.slope_s, self.midpoint, self.normalize)
        elif self.function_curve == "logistic":
            return logistic_saturation(X, self.beta, self.midpoint, self.normalize)
        elif self.function_curve == "exponential":
            return exponential_saturation(X, self.lambda_, self.normalize)
        else:
            raise ValueError("Function curve must be hill, logistic or exponential")


def hill_saturation(x, slope_s, midpoint, normalize=False):
    """
    Parameters:
    X: investimento
    slope_s: Constante de Hill (curvatura)
        - Controla a forma da curva de saturação. Um valor maior que 1 faz com que a curva atinja a saturação mais rapidamente.
    midpoint: ponto de meia saturação (varia de ]0, 1])
        - Este valor representa o gasto em mídia no qual você atinge 50% do efeito máximo (S). Também pode ser ajustado empiricamente ou otimizado.

    Returns:
    array-like: Efeito do gasto em mídia.
    """ 
    if normalize:
        half_saturation_k_transformed = midpoint * (np.max(x) - np.min(x)) + np.min(x)
    else:
        half_saturation_k_transformed = midpoint
        x = np.where(x > 1, 1, x)

    hill = (1 + half_saturation_k_transformed**slope_s / x**slope_s)**-1
    hill = np.nan_to_num(hill, nan = 0)
    return hill

def logistic_saturation(x, beta, midpoint, normalize=False):
    """
    Calcula o efeito de um gasto em mídia usando uma curva logarítmica.

    Parameters:
    x (array-like): Gasto em mídia.
    beta (float): Sensibilidade ao gasto.

    Returns:
    array-like: Efeito do gasto em mídia.
    """
    # if np.min(x) is null, uses 0
    if normalize:
        diff = np.max(x) - np.min(x)
        if diff == 0 or diff is None:
            diff = 1
        normalized_investment = (x - np.min(x)) / diff
    else:
        normalized_investment = np.where(x > 1, 1, x)
    return 1 / (1 + np.exp(-beta * (normalized_investment-midpoint)))

def exponential_saturation(x, lambda_, normalize=False):
    """
    Calcula o efeito de um gasto em mídia usando uma curva exponencial.

    Parameters:
    x (array-like): Gasto em mídia.
    lambda_ (float): Taxa de crescimento do efeito.

    Returns:
    array-like: Efeito do gasto em mídia.
    """
    if normalize:
        diff = np.max(x) - np.min(x)
        if diff == 0 or diff is None:
            diff = 1
        normalized_investment = (x - np.min(x)) / diff
    else:
        normalized_investment = np.where(x > 1, 1, x)
    return (1 - np.exp(-lambda_ * (normalized_investment)))


# if main
if __name__ == "__main__":
    # max_invest = 1000
    max_invest = 1.4
    x = pd.Series(np.linspace(0, max_invest, 21))
    x.index = np.linspace(0, 1, len(x))
    
    plt.figure(figsize=(8, 5))
    plt.plot(x.index, x, label="Original curve")
    plt.ylim(0, 1)
    plt.xlim(0, max_invest)
    plt.title("Efeito da Saturação em diferentes funções")
    plt.xlabel("Investimento")
    plt.ylabel("Retorno")

    # hill
    midpoint = 0.5 # half saturation, range between 0 and 1
    for s in [5]:
        dict_hill = SaturationTransformation(
            function_curve="hill", midpoint=midpoint, slope_s=s,normalize=False
        ).fit_transform(x.values.reshape(-1, 1))
        plt.plot(x, dict_hill, label=f"Hill curve with midpoint = {midpoint} and c = {s}")
    plt.legend()
    plt.show()
    
    # logistic
    for b in [5, 10, 20]:
        log = SaturationTransformation(
            function_curve="logistic", beta=b, midpoint=midpoint
        ).fit_transform(x.values.reshape(-1, 1))
        plt.plot(x, log, label=f"Logistic curve with beta = {b}")
    plt.legend()
    plt.show()
    
    # exp
    for lambda_ in [0.5, 1, 1.5, 2]:
        exp = SaturationTransformation(
            function_curve="exponential", lambda_=lambda_
        ).fit_transform(x.values.reshape(-1, 1))
        plt.plot(x, exp, label=f"Exp curve with lambda = {lambda_}")
        
    plt.legend()
    plt.show()
