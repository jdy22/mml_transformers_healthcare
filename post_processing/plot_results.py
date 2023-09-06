### Functions to create plots of results for the final report

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_dice_scores_CT():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    dice_scores = {
        'Multi-modal model': (92.2, 93.4, 94.1, 77.6, 80.8, 94.5, 84.7, 93.9, 85.7, 76.0, 70.0, 67.4, 69.6, 80.9, 84.6),
        'Baseline model': (89.5, 92.3, 93.3, 70.8, 78.1, 93.4, 79.6, 93.6, 84.9, 72.1, 65.8, 64.0, 65.1, 78.5, 74.3),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(2.5)
    fig.set_figwidth(10)

    for model, score in dice_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('Dice (%)', fontsize=12)
    ax.set_title('Dice score per organ for CT images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=2, fontsize=11)
    ax.set_ylim(50, 100)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_dice_scores_MRI():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    dice_scores = {
        'Multi-modal model': (90.8, 92.3, 93.5, 61.8, 68.8, 95.5, 75.2, 93.8, 84.5, 76.4, 62.9, 66.7, 59.2, 0, 0),
        'Baseline model': (88.1, 90.7, 90.6, 51.8, 65.4, 95.3, 70.4, 93.4, 84.1, 73.0, 53.8, 57.4, 53.3, 0, 0),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(2.5)
    fig.set_figwidth(10)

    for model, score in dice_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('Dice (%)', fontsize=12)
    ax.set_title('Dice score per organ for MRI images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=2, fontsize=11)
    ax.set_ylim(50, 100)
    ax.yaxis.set_ticks(np.arange(50, 101, 10))
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_hd95_scores_CT():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    hd95_scores = {
        'Multi-modal model': (4.6, 3.8, 4.0, 9.3, 3.6, 9.4, 14.6, 2.3, 3.4, 14.1, 4.3, 5.1, 12.6, 9.4, 6.7),
        'Baseline model': (5.5, 4.0, 4.0, 9.5, 3.8, 11.8, 18.5, 2.4, 3.6, 17.3, 4.7, 4.8, 14.7, 10.2, 8.0),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(2.5)
    fig.set_figwidth(10)

    for model, score in hd95_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('HD95 (mm)', fontsize=12)
    ax.set_title('HD95 score per organ for CT images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=2, fontsize=11)
    ax.set_ylim(0, 25)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_hd95_scores_MRI():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    hd95_scores = {
        'Multi-modal model': (6.4, 4.5, 4.6, 11.6, 3.6, 9.8, 13.5, 2.1, 3.2, 10.3, 5.4, 5.3, 17.0, -10, -10),
        'Baseline model': (5.4, 5.4, 4.9, 10.6, 3.7, 10.4, 17.6, 2.2, 3.5, 12.0, 6.2, 5.6, 16.6, -10, -10),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(2.5)
    fig.set_figwidth(10)

    for model, score in hd95_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('HD95 (mm)', fontsize=12)
    ax.set_title('HD95 score per organ for MRI images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=2, fontsize=11)
    ax.set_ylim(0, 25)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_missed_predictions_CT():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    missed_predictions = {
        'Multi-modal model': (1.03, 0.07, 0.07, 1.71, 0.19, 0.05, 0.50, 0.08, 0.23, 1.53, 1.80, 2.15, 1.75, 2.87, 0.56),
        'Baseline model': (3.13, 1.02, 0.53, 9.43, 1.64, 0.51, 2.57, 0.30, 0.57, 3.06, 4.28, 6.50, 5.17, 5.38, 6.18),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(2.5)
    fig.set_figwidth(10)

    for model, score in missed_predictions.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('Missed predictions (%)', fontsize=12)
    ax.set_title('Missed predictions per organ for CT images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=2, fontsize=11)
    ax.set_ylim(0, 12)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_missed_predictions_MRI():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    missed_predictions = {
        'Multi-modal model': (0.81, 0.53, 0.0, 5.96, 3.03, 0.19, 0.53, 0.0, 0.0, 2.24, 0.0, 0.68, 1.38, -10, -10),
        'Baseline model': (2.98, 1.32, 1.10, 23.85, 5.30, 0.29, 4.74, 0.17, 0.27, 4.06, 3.21, 5.86, 3.28, -10, -10),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(4)
    fig.set_figwidth(10)

    for model, score in missed_predictions.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('Missed predictions (%)', fontsize=12)
    ax.set_title('Missed predictions per organ for MRI images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=2, fontsize=11)
    ax.set_ylim(0, 26)
    ax.tick_params(axis='y', labelsize=10)
    ax.yaxis.set_ticks(np.arange(0, 27, 2))

    plt.show()


if __name__ == "__main__":
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:red", "tab:gray"]) 
    # plot_dice_scores_CT()
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["salmon", "tab:gray"]) 
    # plot_dice_scores_MRI()

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:blue", "tab:gray"]) 
    # plot_hd95_scores_CT()
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["cornflowerblue", "tab:gray"]) 
    # plot_hd95_scores_MRI()

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:purple", "tab:gray"]) 
    plot_missed_predictions_CT()
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["plum", "tab:gray"]) 
    plot_missed_predictions_MRI()