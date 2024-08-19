import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
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
        positive_features=[],
        all_features=[],
        biased_features=[],
        normalize_features=[],
        scale_features=[],
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

        self.positive_features = positive_features
        if self.positive_features is None:
            self.positive_features = self.medias
        self.model = None
        self.all_features = all_features
        self.biased_features = biased_features
        self.normalize_features = normalize_features
        self.scale_features = scale_features
        self.confounders = list(set(confounders + self.medias))
        self.best_params = best_params
        self.model = self.initialize_model()

    def transform_to_df(self, X, features):
        return pd.DataFrame(X, columns=features)

    def divide_by_mean(self, X):
        # filter X > 0
        mean_value = X[X > 0].mean()
        mean_value[np.isnan(mean_value)] = 1
        return X / mean_value

    def initialize_model(self):
        self.model = None

        col_sat, col_carry = [], []
        for media in self.medias:
            pipe_carry = Pipeline(
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
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("std_scaler", StandardScaler(), self.scale_features),
                    ("minmax_scaler", MinMaxScaler(), self.normalize_features),
                    # ("divide_by_mean", MeanScaler(), self.medias)
                ],
                remainder="passthrough",  # Manter as outras colunas sem transformação
            )
            pipe_sat = Pipeline(
                [
                    (
                        "saturation",
                        SaturationTransformation(
                            function_curve=self.best_params.get(
                                f"saturation__{media}_pipe__saturation__func", "hill"
                            ),
                            slope_s=self.best_params.get(
                                f"saturation__{media}_saturation_slope_s", 5
                            ),
                            midpoint=self.best_params.get(
                                f"saturation__{media}_pipe__saturation__midpoint", 0.5
                            ),
                            lambda_=self.best_params.get(
                                f"saturation__{media}_pipe__saturation__lambda_", 10
                            ),
                            beta=self.best_params.get(
                                f"saturation__{media}_pipe__saturation__beta", 1
                            ),
                        ),
                    ),
                ]
            )
            col_carry.append((f"{media}_pipe", pipe_carry, [media]))
            col_sat.append((f"{media}_pipe", pipe_sat, [media]))
        adstock = ColumnTransformer(col_carry, remainder="passthrough")
        saturation = ColumnTransformer(col_sat, remainder="passthrough")

        if self.model_type == "linear":
            self.model = Pipeline(
                [
                    ("adstock", adstock),
                    (
                        "to_df1",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={
                                "features": list(
                                    dict.fromkeys(self.medias + self.all_features)
                                )
                            },
                        ),
                    ),
                    ("preprocessor", preprocessor),
                    (
                        "to_df2",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={
                                "features": list(
                                    dict.fromkeys(
                                        self.scale_features
                                        + self.normalize_features
                                        + self.all_features
                                    )
                                )
                            },
                        ),
                    ),
                    ("saturation", saturation),
                    (
                        "to_df3",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={
                                "features": list(
                                    dict.fromkeys(self.medias + self.all_features)
                                )
                            },
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
                        "to_df1",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={
                                "features": list(
                                    dict.fromkeys(self.medias + self.all_features)
                                )
                            },
                        ),
                    ),
                    ("preprocessor", preprocessor),
                    (
                        "to_df2",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={
                                "features": list(
                                    dict.fromkeys(
                                        self.scale_features
                                        + self.normalize_features
                                        + self.all_features
                                    )
                                )
                            },
                        ),
                    ),
                    ("saturation", saturation),
                    (
                        "to_df3",
                        FunctionTransformer(
                            self.transform_to_df,
                            kw_args={
                                "features": list(
                                    dict.fromkeys(self.medias + self.all_features)
                                )
                            },
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

    def predict(self, X):
        check_is_fitted(self, "_is_fitted")
        self._check_n_features(X, reset=False)
        return self.model.predict(X)

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

        joblib.dump(self, dir + "/" + namefile)
        print(f"Model saved in {dir}/{namefile}")

    # static method
    @classmethod
    def load_model(cls, path):
        """Carrega o modelo do disco"""
        cls.model = joblib.load(path)
        cls._is_fitted = True
        return cls.model
