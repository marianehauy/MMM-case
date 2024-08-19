import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array


class MeanScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = check_array(X, ensure_2d=False)
        mean_value = X[X > 0].mean()
        if mean_value is None:
            mean_value = 1
        self.mean_ = mean_value
        return self

    def transform(self, X):
        return X / self.mean_


if __name__ == "__main__":
    X = pd.DataFrame({"investment": [0, 100, 200, 300, 400, 500]})

    mean_scaler = MeanScaler()
    scaled_data = mean_scaler.fit_transform(X)
    print(scaled_data)

    X2 = pd.DataFrame({"investment": [0, 500]})
    scaled_data = mean_scaler.transform(X2)
    print(scaled_data)
