import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin


class ExponentialCarryover(BaseEstimator, TransformerMixin):
    def __init__(self, decay_factor=0.5, L=3, theta=0.5, func="delayed"):
        """
        - decay_factor is the decay factor
        - L is the length of the lag
        - theta is the delay factor
        - func is the function to use for the carryover effect.
            - Options are 'geo' for geometric decay and 'delayed' for delayed adstock
        """
        self.decay_factor = decay_factor
        self.L = L
        self.theta = theta
        self.func = func
        assert self.func in ["geo", "delayed"], "func must be 'geo' or 'delayed'"

    def fit(self, X, y=None):
        # X = check_array(X, ensure_2d=False)
        self._check_n_features(X, reset=True)  # from BaseEstimator

        return self

    def transform(self, X: np.ndarray):
        """
        - X is a vector of media spend going back L timeslots, so it should be len(x) == L
        returns transformed vector of spend
        """
        # check_is_fitted(self)
        # X = check_array(X, ensure_2d=False)
        self._check_n_features(X, reset=False)
        X = np.asarray(X).ravel()

        if self.func == "geo":
            weights = weights_geo(self.decay_factor, self.L)

        elif self.func == "delayed":
            weights = weights_delayed(self.decay_factor, self.theta, self.L)

        result = np.convolve(X, weights, mode="full")
        result = result[: len(X)]

        return pd.DataFrame(result)


def weights_geo(decay_factor, L):
    """
    weighted average with geometric decay
    weight_T = decay_factor ^ T-1
    returns: weights of length L to calculate weighted averages with.
    """
    return decay_factor ** (np.arange(L))


def weights_delayed(decay_factor, theta, L):
    """
    weighted average with delayed adstock function
    weight_T = decay_factor ^ (T-1 - theta)
    returns: weights of length L to calculate weighted averages with.
    """
    return decay_factor ** ((np.arange(L) - theta) ** 2)


def decay_function(half_life):
    return 0.5 ** (1 / half_life)


# if main
if __name__ == "__main__":
    # plot the adstock and the original curve
    # a = pd.Series([100, 0, 0, 0, 50, 0, 0, 20, 40, 0, 0])
    a = pd.Series([1.45, 0, 0, 0, 0, 0, 0, 0, 0])
    L = 5
    decay_factor = 0.50896104498301
    theta = 1
    geo = ExponentialCarryover(decay_factor, L, theta, func="geo").fit_transform(a)
    delayed = ExponentialCarryover(
        decay_factor, L, theta, func="delayed"
    ).fit_transform(a)
    plt.figure(figsize=(15, 5))
    plt.plot(a, label="Original curve")
    plt.plot(geo, label=f"Geometric curve with decay_factor = {decay_factor}")
    plt.plot(
        delayed,
        label=f"Adstocked curve with decay_factor = {decay_factor} and theta = {theta}",
    )
    plt.title("Efeito Adstock em diferentes funções")
    plt.ylabel("Investimento")
    plt.xlabel("Dias")
    plt.legend()
    plt.show()
