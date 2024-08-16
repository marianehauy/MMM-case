import argparse
import json
from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split

# import standart scaller
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.evaluate_model import report_metrics, weighted_absolute_percentage_error
from src.optimize_hyperparams import optimize_linear_regression
from src.prepare_data import add_seasonality, agg_by_week, load_data
from src.regression_model import MMMRegression
from src.utils.plots import plot_prediction, plot_weights

# Features
medias = ["tv", "other_medias", "digital", "offline", "influ", "social"]

season_cols = [
    "year_cos_1",
    "year_sin_1",
    "year_cos_2",
    "year_sin_2",
]
confounders = [
    "n_active_users",
    "notifications_sent",
    "holiday",
    "week_of_month",
    "event1",
    "event2",
]
# all_features = confounders + medias + season_cols
all_features = confounders + medias
scale_features = ["notifications_sent"]
normalize_features = ["n_active_users", "holiday", "week_of_month"]

model_type = "ridge"
scorer = weighted_absolute_percentage_error
scorer = "r2"
n_trials = 300


def main(df=None, mode="train", optimize=False):
    if df is None:
        # load data
        df = (
            load_data()
            .pipe(agg_by_week)
            .pipe(add_seasonality, period_year=52, degree=2, col_name="year")
        )
        assert isinstance(df, pd.DataFrame)

    if mode == "train":
        # train test split
        df_train, df_test = train_test_split(
            df, test_size=0.20, random_state=42, shuffle=False
        )  # shuffle False to keep the order
        print(
            f"Train set from week {int(df_train.index.min())} to {int(df_train.index.max())}"
        )
        print(
            f"Test set from week {int(df_test.index.min())} to {int(df_test.index.max())}"
        )

        norm = MinMaxScaler()
        df_train[normalize_features] = norm.fit_transform(df_train[normalize_features])
        df_test[normalize_features] = norm.transform(df_test[normalize_features])

        scale = StandardScaler()
        df_train[scale_features] = scale.fit_transform(df_train[scale_features])
        df_test[scale_features] = scale.transform(df_test[scale_features])

        if optimize:
            # Load the model
            mmm = MMMRegression(
                medias,
                model_type=model_type,
                positive_features=medias + ["event1", "event2"],
                confounders=confounders,
                biased_features=[],
                all_features=all_features,
            )
            # Optimize with Optuna
            tuned_model = optimize_linear_regression(
                model=mmm.model,
                X=df_train[all_features],
                y=df_train["traffic"],
                medias=medias,
                n_trials=n_trials,
                scorer=scorer,
                dump=False,
                dir="models",
            )
            best_params = tuned_model.best_params_
        else:
            # load json with best hyperparameters from models/hyperparams_calibration.json
            last_file = sorted(
                list(glob("models/hyperparams_calibration_*.json")), reverse=True
            )[0]
            with open(last_file, "r") as f:
                best_params = json.load(f)["best_params"]

        mmm = MMMRegression(
            medias,
            model_type=model_type,
            positive_features=medias + ["event1", "event2"],
            confounders=confounders,
            biased_features=[],
            all_features=all_features,
            best_params=best_params,
        )
        mmm.fit(df_train[all_features], df_train["traffic"])

        pred_train = mmm.predict(df_train[all_features])
        pred_test = mmm.predict(df_test[all_features])
        mmm.score(df_train["traffic"], pred_train)
        mmm.score(df_test["traffic"], pred_test)
        mmm.save_model("models")

        weights = {
            col: float(coef)
            for col, coef in zip(
                all_features, mmm.model.named_steps["regression"].coef_
            )
        }
        weights = pd.Series(weights).sort_values(ascending=False)
        plot_weights(weights / weights.sum() * 100, figsize=(10, 5), save=True)

        plot_prediction(
            df_train["traffic"],
            df_test["traffic"],
            pred_train,
            pred_test,
            fig_size=(10, 5),
            save=True,
        )
    elif mode == "predict":
        norm = MinMaxScaler()
        df[normalize_features] = norm.fit_transform(df[normalize_features])

        scale = StandardScaler()
        df[scale_features] = scale.fit_transform(df[scale_features])
        # Load the model
        path = sorted(list(glob("models/regression_model_*.joblib")), reverse=True)[0]
        mmm = MMMRegression(
            medias,
            model_type=model_type,
            positive_features=medias + ["event1", "event2"],
            confounders=confounders,
            biased_features=[],
            all_features=all_features,
        ).load_model(path=path)
        mmm.predict(df[all_features])
        report_metrics(df["traffic"], mmm.predict(df[all_features]))

    else:
        raise ValueError("Mode should be either 'train' or 'predict'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="Mode to run the model",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help="Whether to optimize the model",
    )

    args = parser.parse_args()
    main(args)

    main(mode="train", optimize=False)
    # main(mode="train", optimize=True)
    # main(mode="predict", optimize=False)
