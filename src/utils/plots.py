import calendar

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme()
calendar.setfirstweekday(0)


def plot_kpis(df, plot_dict) -> None:
    """
    Plot KPIs for each month
    Parameters:
    - df: DataFrame
    - plot_dict: dictionary
    """
    # create n subplots
    fig, axs = plt.subplots(len(plot_dict), 1, figsize=(12, len(plot_dict) * 2.5))
    # create space between subplots
    plt.subplots_adjust(hspace=0.4)

    for i, (col, title) in enumerate(plot_dict.items()):
        df[col].plot(ax=axs[i])
        axs[i].title.set_text(title)
        if i != len(plot_dict):
            axs[i].set_xlabel(None)


def plot_histograms_features(df, features):
    nrows = len(features)
    fig, axs = plt.subplots(
        nrows=nrows, ncols=2, figsize=(12, 2.5 * nrows), layout="tight"
    )

    for idx, feat in enumerate(features):
        sns.histplot(data=df, x=feat, kde=True, ax=axs[idx, 0], color="navy")
        axs[idx, 0].set_title(f"{feat} Histogram")
        axs[idx, 0].axvline(x=df[feat].mean(), color="red", linestyle="--")
        sns.boxplot(data=df, x=feat, ax=axs[idx, 1], color="crimson")
        axs[idx, 1].set_title(f"{feat} Boxplot")

    plt.show()


def plot_corr_heatmap(df, figsize=(15, 7)):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap="OrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.33,
        annot_kws={"fontsize": "x-small"},
        mask=mask,
    )
    plt.title("Correlation Matrix")


def plot_media_traffic(df, medias, fig_size=(15, 5)):
    order_cols = df[medias].sum().sort_values(ascending=False).index
    df[order_cols].plot.bar(
        stacked=True, figsize=fig_size, secondary_y=False, label="Media Investment"
    ).legend(loc="best")
    df["traffic"].plot(secondary_y=True, label="Traffic on website")
    plt.fill_between(
        df.index,
        df["traffic"],
        where=df["event1"],
        color="orange",
        alpha=0.25,
        label="Event 1",
    )
    plt.fill_between(
        df.index,
        df["traffic"],
        where=df["event2"],
        color="green",
        alpha=0.25,
        label="Event2",
    )
    plt.legend(loc="center left", bbox_to_anchor=(1.03, 0.5))
    plt.title("Total Media investment x Total Traffic on website")
    plt.show()


def plot_stacked_media(df, medias, fig_size=(15, 5)):
    # order by highest investment
    df_plot = df[medias].div(df[medias].sum(axis=1), axis=0)
    order_cols = df_plot.sum().sort_values(ascending=False).index
    df_plot[order_cols].plot(kind="bar", stacked=True, figsize=fig_size)
    plt.title("Media distribution by month")
    plt.ylabel("Percentage")
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.show()


def plot_prediction(
    y_train: pd.Series,
    y_test,
    y_train_pred,
    y_test_pred,
    fig_size=(15, 5),
    save=False,
    save_path="reports/plots/prediction.png",
):
    plt.figure(figsize=fig_size)

    ax = plt.plot(y_train.index, y_train, label="Train Data")
    ax = plt.plot(y_train.index, y_train_pred, label="Train Prediction")
    ax = plt.plot(y_test.index, y_test, label="Test Data")
    ax = plt.plot(y_test.index, y_test_pred, label="Test Prediction")
    plt.title("Train and Test Prediction")
    plt.xlabel("Weeks")
    plt.ylabel("Traffic")
    plt.legend()
    plt.show()

    if save:
        ax.figure.savefig(save_path)
        print("Media weights saved as reports/plots/media_weights.png")


def plot_weights(
    media_weights,
    figsize=(7, 4),
    save=False,
    save_path="reports/plots/media_weights.png",
):
    ax = media_weights.sort_values(ascending=False).plot.bar(
        color="darkred", figsize=figsize
    )
    plt.title("Media weights", fontsize=20)
    plt.ylabel("Weight (%)")
    plt.xticks(rotation=45)
    if save:
        ax.figure.savefig(save_path)
        print("Media weights saved as reports/plots/media_weights.png")
    plt.show()
