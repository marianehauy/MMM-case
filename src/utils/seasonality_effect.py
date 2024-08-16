import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# write seasonality function


def get_seasonality(date_series, period=52, degree=2):
    date_series = np.arange(0, len(date_series))
    df_seas = pd.DataFrame()
    for i in np.arange(1, degree + 1):
        df_seas[f"cos_{i}"] = np.cos(2 * np.pi * i * date_series / period)
        df_seas[f"sin_{i}"] = np.sin(2 * np.pi * i * date_series / period)
    return df_seas


# if main
if __name__ == "__main__":
    # plot the adstock and the original curve
    teste = pd.DataFrame({"t": np.arange(0, 52 * 2)})

    # plot the seasonality when period = 4 and degree = 2 and when period = 52 and degree = 2
    plt.figure(figsize=(15, 5))
    plt.plot(
        get_seasonality(teste["t"], period=4, degree=2).iloc[:, 0],
        label="Period = 4 and Degree = 2",
    )
    plt.plot(
        get_seasonality(teste["t"], period=52, degree=2).iloc[:, 0],
        label="Period = 52 and Degree = 2",
    )
    plt.title("Representação da sazonalidade com Fourrier")
    plt.xlabel("Semanas")
    plt.legend()
    plt.show()
