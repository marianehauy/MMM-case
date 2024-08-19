import json
import warnings

import pandas as pd
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.integration import OptunaSearchCV
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

from src.evaluate_model import weighted_absolute_percentage_error

warnings.filterwarnings("ignore")

def optimize_linear_regression(
    model,
    X,
    y,
    medias,
    n_trials=500,
    scorer=weighted_absolute_percentage_error,
    #
    dump=False,
    dir="..models",
):
    param_distributions = {}
    for media in medias:
        # ADSTOCK
        param_distributions[
            f"adstock__{media}_pipe__carryover__func"
        ] = CategoricalDistribution([
            # "geo", 
            "delayed"
            ])
        # decay factor
        param_distributions[
            f"adstock__{media}_pipe__carryover__decay_factor"
        ] = FloatDistribution(0.1, 1)
        # Lag
        param_distributions[f"adstock__{media}_pipe__carryover__L"] = IntDistribution(
            3, 8
        )
        # theta
        param_distributions[
            f"adstock__{media}_pipe__carryover__theta"
        ] = IntDistribution(0, 3)

        # SATURATION
        # saturation__func can be "hill", "log" or "exponential"
        param_distributions[
            f"saturation__{media}_pipe__saturation__function_curve"
        ] = CategoricalDistribution([
            "hill", 
            # "logistic", 
            # "exponential"
            ])
        
        param_distributions[
            f"saturation__{media}_pipe__saturation__midpoint"
        ] = FloatDistribution(
            0.1, 1
        )

        param_distributions[
            f"saturation__{media}_pipe__saturation__slope_s"
        ] = FloatDistribution(0.5, 10)

        # param_distributions[
        #     f"saturation__{media}_pipe__saturation__beta"
        # ] = FloatDistribution(5, 20)

        # param_distributions[
        #     f"saturation__{media}_pipe__saturation__lambda_"
        # ] = FloatDistribution(0.5, 2)        


        # write the distribution for the regression
        param_distributions["regression__alpha"] = FloatDistribution(0.2, 1)

    if not isinstance(scorer, str):
        scorer = make_scorer(score_func=scorer)
    # turnoff verbose

    tuned_model = OptunaSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_trials=n_trials,
        cv=TimeSeriesSplit(n_splits=3, test_size=14),
        random_state=0,
        scoring=scorer,
        n_jobs=-1,
    )
    tuned_model.fit(X, y)

    # Melhores parâmetros e pontuação
    print("Melhores parâmetros encontrados: ", tuned_model.best_params_)
    print("Melhor Score: ", tuned_model.best_score_)
    # save best_params and best score in a JSON file
    if dump:
        date_suffix = pd.Timestamp.now().strftime("%Y%m%d")
        with open(f"{dir}/hyperparams_calibration_{date_suffix}.json", "w") as f:
            json.dump(
                {
                    "best_params": tuned_model.best_params_,
                    "wape": -tuned_model.best_score_,
                },
                f,
            )
        print(
            f"Hyperparameters saved in ../models/hyperparams_calibration_{date_suffix}.json"
        )
    return tuned_model
