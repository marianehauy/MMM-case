import joblib
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_is_fitted

from src.evaluate_model import report_metrics
from src.utils.carryover_effect import ExponentialCarryover
from src.utils.orthogonalization import VariableOrthogonalization
from src.utils.ridge_with_constrains_model import RidgeWithPositiveConstraints
from src.utils.saturation_effect import SaturationTransformation


class MMMRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        medias,
        model_type="linear",
        positive_features=None,
        all_features=None,
        biased_features=[],
        confounders=[
            "n_active_users",
            "notifications_sent",
            "holiday",
            "event1",
            "event2",
        ],
        best_params={"a": 1},
    ):
        self.medias = medias
        self.model_type = model_type
        self.col_transf = []
        self.positive_features = positive_features
        if self.positive_features is None:
            self.positive_features = self.medias
        self.model = None
        self.all_features = all_features
        self.biased_features = biased_features
        self.confounders = list(set(confounders + self.medias))
        self.best_params = best_params
        self.model = self.initialize_model()

    def transform_to_df(self, X, features):
        return pd.DataFrame(X, columns=features)

    def initialize_model(self):
        self.model = None

        # transformation by media
        for media in self.medias:
            pipe = Pipeline(
                [
                    (
                        "carryover",
                        ExponentialCarryover(
                            decay_factor=self.best_params.get(
                                f"adstock__{media}_pipe__carryover__decay_factor", 0.8
                            ),
                            L=self.best_params.get(
                                f"adstock__{media}_pipe__carryover__L", 4
                            ),
                            theta=self.best_params.get(
                                f"adstock__{media}_pipe__carryover__theta", 1
                            ),
                            func=self.best_params.get(
                                f"adstock__{media}_pipe__carryover__func", "geo"
                            ),
                        ),
                    ),
                    (
                        "saturation",
                        SaturationTransformation(
                            c=self.best_params.get(
                                f"adstock__{media}_saturation_c", 10
                            ),
                            midpoint=self.best_params.get(
                                f"adstock__{media}_pipe__saturation__midpoint", None
                            ),
                            lambda_=self.best_params.get(
                                f"adstock__{media}_pipe__saturation__lambda_", 0.00001
                            ),
                            function_curve=self.best_params.get(
                                f"adstock__{media}_pipe__saturation__func", "log"
                            ),
                        ),
                    ),
                ]
            )
            self.col_transf.append((f"{media}_pipe", pipe, [media]))
        adstock = ColumnTransformer(self.col_transf, remainder="passthrough")

        if self.model_type == "linear":
            self.model = Pipeline(
                [
                    ("adstock", adstock),
                    (
                        "to_df",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={"features": self.all_features},
                        ),
                    ),
                    (
                        "deorth",
                        VariableOrthogonalization(
                            self.biased_features, confounders=self.confounders
                        ),
                    ),
                    ("regression", LinearRegression(positive=True)),
                ]
            )
        elif self.model_type == "ridge":
            self.positive_features = [
                i
                for i, feat in enumerate(self.all_features)
                if feat in self.positive_features
            ]
            self.model = Pipeline(
                [
                    ("adstock", adstock),
                    (
                        "to_df",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={"features": self.all_features},
                        ),
                    ),
                    (
                        "deorth",
                        VariableOrthogonalization(
                            self.biased_features, confounders=self.confounders
                        ),
                    ),
                    (
                        "regression",
                        RidgeWithPositiveConstraints(
                            alpha=self.best_params.get("regression__alpha", 0.2),
                            positive_features=self.positive_features,
                        ),
                    ),
                ]
            )
        else:
            raise ValueError(
                "Model type not recognized. Choose between 'linear' or 'ridge'"
            )

        return self.model

    def fit(self, X, y=None):
        # X = check_array(X, ensure_2d=False)
        self._check_n_features(X, reset=True)  # from BaseEstimator
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict(self, df):
        check_is_fitted(self, "_is_fitted")
        self._check_n_features(df, reset=False)
        return self.model.predict(df)

    def score(self, y_true, y_pred):
        # y_true = check_array(y_true, ensure_2d=False)
        # y_pred = check_array(y_pred, ensure_2d=False)
        return report_metrics(y_true, y_pred)

    def set_best_params(self, best_params):
        """Método para aplicar os hiperparâmetros otimizados"""
        self.best_params = best_params

    # load and save model
    def save_model(self, dir="../models/"):
        """Salva o modelo em disco"""
        date_suffix = pd.Timestamp.now().strftime("%Y%m%d")
        namefile = f"regression_model_{date_suffix}.joblib"

        joblib.dump(self.model, dir + "/" + namefile)
        print(f"Model saved in {dir}/{namefile}")

    # static method
    @classmethod
    def load_model(cls, path):
        """Carrega o modelo do disco"""
        cls.model = joblib.load(path)
        cls._is_fitted = True
        return cls.model
