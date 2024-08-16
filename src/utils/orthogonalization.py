import statsmodels.formula.api as smf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class VariableOrthogonalization(BaseEstimator, TransformerMixin):
    def __init__(self, variable_orth, confounders):
        self.variable_orth = variable_orth
        if isinstance(variable_orth, str):
            self.variable_lst = [variable_orth]
        elif isinstance(variable_orth, list):
            self.variable_lst = variable_orth
        else:
            raise ValueError("Confounders must be a string or a list")
        self.confounders = confounders
        self.model = {}

    def fit(self, X, y=None):
        # X = check_array(X, )
        self._check_n_features(X, reset=False)  # from BaseEstimator
        for variable_str in self.variable_lst:
            confounders_str = " + ".join(
                [c for c in self.confounders if c != variable_str]
            )
            self.model[variable_str] = smf.ols(
                f"{variable_str} ~ {confounders_str}", data=X
            ).fit()
        self._is_fitted = True
        return self

    def transform(self, X):
        check_is_fitted(self, "_is_fitted")
        # X = check_array(X)
        self._check_n_features(X, reset=False)  # from BaseEstimator
        for variable_str in self.variable_lst:
            # X[f"variable_str_backup"] = X[variable_str]
            X[variable_str] = X[variable_str] - self.model[variable_str].predict(X)
        return X


# if main
if __name__ == "__main__":
    pass
