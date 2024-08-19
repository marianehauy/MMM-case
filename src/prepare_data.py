import os

import pandas as pd

from src.utils.seasonality_effect import get_seasonality
from src.utils.utils import week_of_month

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__name__))


def load_data(current_dir=current_dir):
    df = pd.read_csv(current_dir + "/data/raw/mmm_dataset.csv")

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["month"] = df.date.dt.strftime("%Y-%m")
    df["weekday"] = df.date.dt.weekday
    # day before holiday
    df["day_before_holiday"] = df["holiday"].shift(-1).fillna(0)

    return df


def agg_by_week(df):
    # agregate by week_from_start, and considers the sum of metrics and the max of n_active users
    cols_agg = {
        "traffic": "sum",
        "notifications_sent": "sum",
        "n_active_users": "max",  # or mean
        "event1": "max",
        "event2": "max",
        "cinema": "sum",
        "digital": "sum",
        "influ": "sum",
        "newpapers": "sum",
        "offline": "sum",
        "radio": "sum",
        "social": "sum",
        "tv": "sum",
        "other_medias": "sum",
        "holiday": "sum",  # assumption: more holidays, more or less traffic than if 1 holiday
        "month": "max",
    }

    # usar resample para agrupar por semana, considerando a semana de segunda a domingo
    df_week = df.set_index("date").resample("W-Mon").agg(cols_agg).reset_index()
    df_week["week_of_month"] = df_week.date.apply(week_of_month)
    # remove uncomplete weeks: remove first and last week
    df_week = df_week.iloc[1:-1]
    df_week = df_week.reset_index(drop=False)
    df_week = df_week.rename({"index": "weeks_from_start"}, axis=1)

    dt_first_major_investment = pd.to_datetime("2007-05-01")
    df_week["has_media_investment"] = df_week.date >= dt_first_major_investment
    return df_week


def add_seasonality(df, period_year=52, degree=2, col_name="year"):
    df_season_year = get_seasonality(df["date"], period=period_year, degree=degree)
    df_season_year.columns = [col_name + "_" + col for col in df_season_year.columns]
    df = pd.concat([df, df_season_year], axis=1)
    return df


if __name__ == "__main__":
    df = (
        load_data()
        .pipe(agg_by_week)
        .pipe(add_seasonality, period_year=52, degree=2, col_name="year")
    )
    assert isinstance(df, pd.DataFrame)
