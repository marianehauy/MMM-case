import calendar

import numpy as np
import pandas as pd
from scipy.stats import shapiro
calendar.setfirstweekday(0)

def week_of_month(dt: str | pd.Timestamp) -> int:
    """
    Get the week of the month for a given date
    Parameters:
    - dt: datetime or string
    """
    if isinstance(dt, str):
        dt = pd.to_datetime(dt)
    x = np.array(calendar.monthcalendar(dt.year, dt.month))
    week_of_month = np.where(x == dt.day)[0][0]
    if week_of_month == 0:
        return 1
    return week_of_month


def is_normal(series, verbose=True):
    # Shapiro-Wilk
    _, p_value = shapiro(series)
    # p-value < 0.05, so we reject the null hypothesis that the data is normally distributed
    if p_value < 0.05:
        if verbose:
            print("Data is not normally distributed")
        return False
    else:
        if verbose:
            print("Data can be considered normally distributed")
        return True
