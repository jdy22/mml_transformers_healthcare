### Functions to create plots of results for the final report

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_dice_scores_CT():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    dice_scores = {
        'Best-performing context-aware model': (92.2, 93.4, 94.1, 77.6, 80.8, 94.5, 84.7, 93.9, 85.7, 76.0, 70.0, 67.4, 69.6, 80.9, 84.6),
        'Baseline model': (89.5, 92.3, 93.3, 70.8, 78.1, 93.4, 79.6, 93.6, 84.9, 72.1, 65.8, 64.0, 65.1, 78.5, 74.3),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(3)
    fig.set_figwidth(10)

    for model, score in dice_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('Dice (%)', fontsize=12)
    ax.set_title('Dice score per organ for CT images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=1, fontsize=10)
    ax.set_ylim(50, 100)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_dice_scores_MRI():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    dice_scores = {
        'Best-performing context-aware model': (90.8, 92.3, 93.5, 61.8, 68.8, 95.5, 75.2, 93.8, 84.5, 76.4, 62.9, 66.7, 59.2, 0, 0),
        'Baseline model': (88.1, 90.7, 90.6, 51.8, 65.4, 95.3, 70.4, 93.4, 84.1, 73.0, 53.8, 57.4, 53.3, 0, 0),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(3)
    fig.set_figwidth(10)

    for model, score in dice_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('Dice (%)', fontsize=12)
    ax.set_title('Dice score per organ for MRI images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=1, fontsize=10)
    ax.set_ylim(50, 100)
    ax.yaxis.set_ticks(np.arange(50, 101, 10))
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_hd95_scores_CT():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    hd95_scores = {
        'Best-performing context-aware model': (4.6, 3.8, 4.0, 9.3, 3.6, 9.4, 14.6, 2.3, 3.4, 14.1, 4.3, 5.1, 12.6, 9.4, 6.7),
        'Baseline model': (5.5, 4.0, 4.0, 9.5, 3.8, 11.8, 18.5, 2.4, 3.6, 17.3, 4.7, 4.8, 14.7, 10.2, 8.0),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(3)
    fig.set_figwidth(10)

    for model, score in hd95_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('HD95 (mm)', fontsize=12)
    ax.set_title('HD95 score per organ for CT images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=1, fontsize=10)
    ax.set_ylim(0, 25)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_hd95_scores_MRI():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    hd95_scores = {
        'Best-performing context-aware model': (6.4, 4.5, 4.6, 11.6, 3.6, 9.8, 13.5, 2.1, 3.2, 10.3, 5.4, 5.3, 17.0, -10, -10),
        'Baseline model': (5.4, 5.4, 4.9, 10.6, 3.7, 10.4, 17.6, 2.2, 3.5, 12.0, 6.2, 5.6, 16.6, -10, -10),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(3)
    fig.set_figwidth(10)

    for model, score in hd95_scores.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('HD95 (mm)', fontsize=12)
    ax.set_title('HD95 score per organ for MRI images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=1, fontsize=10)
    ax.set_ylim(0, 25)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_missed_predictions_CT():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    missed_predictions = {
        'Best-performing context-aware model': (1.03, 0.07, 0.07, 1.71, 0.19, 0.05, 0.50, 0.08, 0.23, 1.53, 1.80, 2.15, 1.75, 2.87, 0.56),
        'Baseline model': (3.13, 1.02, 0.53, 9.43, 1.64, 0.51, 2.57, 0.30, 0.57, 3.06, 4.28, 6.50, 5.17, 5.38, 6.18),
    }

    x = np.arange(len(organs))  # the label locations
    width = 0.425  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')
    fig.set_figheight(3)
    fig.set_figwidth(10)

    for model, score in missed_predictions.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, score, width, label=model)
        ax.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax.set_ylabel('Missed predictions (%)', fontsize=12)
    ax.set_title('Missed predictions per organ for CT images', fontsize=12)
    ax.set_xticks(x + 0.5*width, organs, fontsize=10)
    ax.legend(loc='upper right', ncols=1, fontsize=10)
    ax.set_ylim(0, 12)
    ax.tick_params(axis='y', labelsize=10)

    plt.show()


def plot_missed_predictions_MRI():
    organs = ("Spl", "RK", "LK", "Gall", "Eso", "Liv", "Stom", "Aorta", "Post", "Pan", "RAG", "LAG", "Duod", "Blad", "Pros/uter")
    missed_predictions = {
        'Best-performing context-aware model': (0.81, 0.53, 0.0, 5.96, 3.03, 0.19, 0.53, 0.0, 0.0, 2.24, 0.0, 0.68, 1.38, -10, -10),
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
    ax.legend(loc='upper right', ncols=1, fontsize=10)
    ax.set_ylim(0, 26)
    ax.tick_params(axis='y', labelsize=10)
    ax.yaxis.set_ticks(np.arange(0, 27, 2))

    plt.show()


def plot_type_results():
    scores_dice = ("CT", "MRI")
    values_dice = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (83.1, 78.5),
        'Early concatenation of image modality information with learnable embeddings (method 6b)': (79.6, 74.9),
        'Baseline model': (79.7, 74.4),
    }

    scores_hd95 = ("CT", "MRI")
    values_hd95 = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (7.1, 7.5),
        'Early concatenation of image modality information with learnable embeddings (method 6b)': (8.3, 8.2),
        'Baseline model': (8.2, 8.0),
    }

    scores_MP = ("CT", "MRI")
    values_MP = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (0.97, 1.18),
        'Early concatenation of image modality information with learnable embeddings (method 6b)': (3.32, 3.99),
        'Baseline model': (3.35, 4.34),
    }

    x = np.arange(2)  # the label locations
    width = 0.25  # the width of the bars

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained')
    fig.set_figheight(4)
    fig.set_figwidth(10)

    multiplier = 0
    for model, value in values_dice.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, value, width, label=model)
        ax1.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_hd95.items():
        offset = width * multiplier
        rects = ax2.bar(x + offset, value, width, label="_")
        ax2.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_MP.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, value, width, label="_")
        ax3.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax1.set_xticks(x + 1*width, scores_dice, fontsize=10)
    ax2.set_xticks(x + 1*width, scores_hd95, fontsize=10)
    ax3.set_xticks(x + 1*width, scores_MP, fontsize=10)
    ax1.set_xlabel('mDice (%)', fontsize=11)
    ax2.set_xlabel('mHD95 (mm)', fontsize=11)
    ax3.set_xlabel('Missed predictions (%)', fontsize=11)
    ax1.set_ylim(70, 85)
    ax2.set_ylim(7, 8.5)
    ax3.set_ylim(0, 5)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width*1.1, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width*1.1, box2.height*0.9])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width*1.1, box3.height*0.9])

    fig.legend(bbox_to_anchor=(0.525,1.0), loc='upper center', ncols=1, fontsize=10)

    plt.show()


def plot_embeddings_results():
    scores_dice = ("CT", "MRI")
    values_dice = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (83.1, 78.5),
        'Early concatenation of organ information with CLIP embeddings (method 2)': (82.9, 78.7),
        'Baseline model': (79.7, 74.4),
    }

    scores_hd95 = ("CT", "MRI")
    values_hd95 = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (7.1, 7.5),
        'Early concatenation of organ information with CLIP embeddings (method 2)': (7.2, 7.4),
        'Baseline model': (8.2, 8.0),
    }

    scores_MP = ("CT", "MRI")
    values_MP = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (0.97, 1.18),
        'Early concatenation of organ information with CLIP embeddings (method 2)': (1.03, 1.34),
        'Baseline model': (3.35, 4.34),
    }

    x = np.arange(2)  # the label locations
    width = 0.25  # the width of the bars

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained')
    fig.set_figheight(4)
    fig.set_figwidth(10)

    multiplier = 0
    for model, value in values_dice.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, value, width, label=model)
        ax1.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_hd95.items():
        offset = width * multiplier
        rects = ax2.bar(x + offset, value, width, label="_")
        ax2.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_MP.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, value, width, label="_")
        ax3.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax1.set_xticks(x + 1*width, scores_dice, fontsize=10)
    ax2.set_xticks(x + 1*width, scores_hd95, fontsize=10)
    ax3.set_xticks(x + 1*width, scores_MP, fontsize=10)
    ax1.set_xlabel('mDice (%)', fontsize=11)
    ax2.set_xlabel('mHD95 (mm)', fontsize=11)
    ax3.set_xlabel('Missed predictions (%)', fontsize=11)
    ax1.set_ylim(70, 85)
    ax2.set_ylim(7, 8.5)
    ax3.set_ylim(0, 5)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width*1.1, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width*1.1, box2.height*0.9])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width*1.1, box3.height*0.9])

    fig.legend(bbox_to_anchor=(0.525,1.0), loc='upper center', ncols=1, fontsize=10)

    plt.show()


def plot_method_results1():
    scores_dice = ("CT", "MRI")
    values_dice = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (83.1, 78.5),
        'Intermediate concatenation of organ information with learnable embeddings (method 3a)': (80.9, 76.5),
        'Intermediate concatenation of organ information with learnable embeddings (method 3b)': (81.3, 76.4),
        'Intermediate concatenation of organ information with learnable embeddings (method 3c)': (81.2, 76.1),
        'Late concatenation of organ information with learnable embeddings (method 4)': (80.2, 75.2),
        'CLIP-driven inspired fusion of organ information (method 5)': (81.6, 76.5),
        'Baseline model': (79.7, 74.4),
    }

    scores_hd95 = ("CT", "MRI")
    values_hd95 = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (7.1, 7.5),
        'Intermediate concatenation of organ information with learnable embeddings (method 3a)': (7.6, 7.5),
        'Intermediate concatenation of organ information with learnable embeddings (method 3b)': (7.4, 7.5),
        'Intermediate concatenation of organ information with learnable embeddings (method 3c)': (7.4, 7.7),
        'Late concatenation of organ information with learnable embeddings (method 4)': (8.0, 8.3),
        'CLIP-driven inspired fusion of organ information (method 5)': (7.3, 7.5),
        'Baseline model': (8.2, 8.0),
    }

    scores_MP = ("CT", "MRI")
    values_MP = {
        'Early concatenation of organ information with learnable embeddings (method 1)': (0.97, 1.18),
        'Intermediate concatenation of organ information with learnable embeddings (method 3a)': (2.06, 2.64),
        'Intermediate concatenation of organ information with learnable embeddings (method 3b)': (2.03, 2.54),
        'Intermediate concatenation of organ information with learnable embeddings (method 3c)': (2.10, 2.86),
        'Late concatenation of organ information with learnable embeddings (method 4)': (2.63, 3.44),
        'CLIP-driven inspired fusion of organ information (method 5)': (1.51, 2.48),
        'Baseline model': (3.35, 4.34),
    }

    x = np.arange(2)  # the label locations
    width = 0.13  # the width of the bars

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained')
    fig.set_figheight(6.5)
    fig.set_figwidth(16)

    multiplier = 0
    for model, value in values_dice.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, value, width, label=model)
        ax1.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.0)
        multiplier += 1

    multiplier = 0
    for model, value in values_hd95.items():
        offset = width * multiplier
        rects = ax2.bar(x + offset, value, width, label="_")
        ax2.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.0)
        multiplier += 1

    multiplier = 0
    for model, value in values_MP.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, value, width, label="_")
        ax3.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.0)
        multiplier += 1

    ax1.set_xticks(x + 3*width, scores_dice, fontsize=14)
    ax2.set_xticks(x + 3*width, scores_hd95, fontsize=14)
    ax3.set_xticks(x + 3*width, scores_MP, fontsize=14)
    ax1.set_xlabel('mDice (%)', fontsize=15)
    ax2.set_xlabel('mHD95 (mm)', fontsize=15)
    ax3.set_xlabel('Missed predictions (%)', fontsize=15)
    ax1.set_ylim(70, 85)
    ax2.set_ylim(7, 8.5)
    ax3.set_ylim(0, 5)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width*1.1, box1.height*0.7])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width*1.1, box2.height*0.7])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width*1.1, box3.height*0.7])

    fig.legend(bbox_to_anchor=(0.525,1.0), loc='upper center', ncols=1, fontsize=14)

    plt.show()


def plot_method_results2():
    scores_dice = ("CT", "MRI")
    values_dice = {
        'Early concatenation of image modality information with learnable embeddings (method 6a)': (79.5, 74.8),
        'Early concatenation of image modality information with learnable embeddings (method 6b)': (79.6, 74.9),
        'Early summation of image modality information with learnable embeddings (method 6c)': (79.7, 74.5),
        'Baseline model': (79.7, 74.4),
    }

    scores_hd95 = ("CT", "MRI")
    values_hd95 = {
        'Early concatenation of image modality information with learnable embeddings (method 6a)': (8.2, 8.2),
        'Early concatenation of image modality information with learnable embeddings (method 6b)': (8.3, 8.2),
        'Early summation of image modality information with learnable embeddings (method 6c)': (8.2, 8.2),
        'Baseline model': (8.2, 8.0),
    }

    scores_MP = ("CT", "MRI")
    values_MP = {
        'Early concatenation of image modality information with learnable embeddings (method 6a)': (3.32, 4.07),
        'Early concatenation of image modality information with learnable embeddings (method 6b)': (3.32, 3.99),
        'Early summation of image modality information with learnable embeddings (method 6c)': (3.28, 4.31),
        'Baseline model': (3.35, 4.34),
    }

    x = np.arange(2)  # the label locations
    width = 0.2  # the width of the bars

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained')
    fig.set_figheight(4)
    fig.set_figwidth(10)

    multiplier = 0
    for model, value in values_dice.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, value, width, label=model)
        ax1.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_hd95.items():
        offset = width * multiplier
        rects = ax2.bar(x + offset, value, width, label="_")
        ax2.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_MP.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, value, width, label="_")
        ax3.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax1.set_xticks(x + 1.5*width, scores_dice, fontsize=10)
    ax2.set_xticks(x + 1.5*width, scores_hd95, fontsize=10)
    ax3.set_xticks(x + 1.5*width, scores_MP, fontsize=10)
    ax1.set_xlabel('mDice (%)', fontsize=11)
    ax2.set_xlabel('mHD95 (mm)', fontsize=11)
    ax3.set_xlabel('Missed predictions (%)', fontsize=11)
    ax1.set_ylim(72, 82)
    ax2.set_ylim(7.8, 8.5)
    ax3.set_ylim(3, 5)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width*1.1, box1.height*0.82])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width*1.1, box2.height*0.82])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width*1.1, box3.height*0.82])

    fig.legend(bbox_to_anchor=(0.525,1.0), loc='upper center', ncols=1, fontsize=10)

    plt.show()


def plot_method_results3():
    scores_dice = ("CT", "MRI")
    values_dice = {
        'Separate decoders and output layers for each image modality, trained from scratch (method 7a)': (79.4, 68.7),
        'Separate decoders and output layers for each image modality, fine-tuned from baseline model (method 7b)': (79.7, 74.2),
        'Baseline model': (79.7, 74.4),
    }

    scores_hd95 = ("CT", "MRI")
    values_hd95 = {
        'Separate decoders and output layers for each image modality, trained from scratch (method 7a)': (8.4, 10.1),
        'Separate decoders and output layers for each image modality, fine-tuned from baseline model (method 7b)': (8.5, 8.7),
        'Baseline model': (8.2, 8.0),
    }

    scores_MP = ("CT", "MRI")
    values_MP = {
        'Separate decoders and output layers for each image modality, trained from scratch (method 7a)': (3.17, 7.29),
        'Separate decoders and output layers for each image modality, fine-tuned from baseline model (method 7b)': (2.98, 4.50),
        'Baseline model': (3.35, 4.34),
    }

    x = np.arange(2)  # the label locations
    width = 0.25  # the width of the bars

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained')
    fig.set_figheight(4)
    fig.set_figwidth(10)

    multiplier = 0
    for model, value in values_dice.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, value, width, label=model)
        ax1.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_hd95.items():
        offset = width * multiplier
        rects = ax2.bar(x + offset, value, width, label="_")
        ax2.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_MP.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, value, width, label="_")
        ax3.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax1.set_xticks(x + 1*width, scores_dice, fontsize=10)
    ax2.set_xticks(x + 1*width, scores_hd95, fontsize=10)
    ax3.set_xticks(x + 1*width, scores_MP, fontsize=10)
    ax1.set_xlabel('mDice (%)', fontsize=11)
    ax2.set_xlabel('mHD95 (mm)', fontsize=11)
    ax3.set_xlabel('Missed predictions (%)', fontsize=11)
    ax1.set_ylim(65, 85)
    ax2.set_ylim(7, 10.5)
    ax3.set_ylim(0, 9)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width*1.1, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width*1.1, box2.height*0.9])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width*1.1, box3.height*0.9])

    fig.legend(bbox_to_anchor=(0.525,1.0), loc='upper center', ncols=1, fontsize=10)

    plt.show()


def plot_method_results4():
    scores_dice = ("CT", "MRI")
    values_dice = {
        'Joint classification and segmentation model (method 8)': (78.7, 74.3),
        'Intermediate concatenation of organ information with learnable embeddings (method 3c)': (81.2, 76.1),
        'Baseline model': (79.7, 74.4),
    }

    scores_hd95 = ("CT", "MRI")
    values_hd95 = {
        'Joint classification and segmentation model (method 8)': (8.6, 8.7),
        'Intermediate concatenation of organ information with learnable embeddings (method 3c)': (7.4, 7.7),
        'Baseline model': (8.2, 8.0),
    }

    scores_MP = ("CT", "MRI")
    values_MP = {
        'Joint classification and segmentation model (method 8)': (3.35, 3.96),
        'Intermediate concatenation of organ information with learnable embeddings (method 3c)': (2.10, 2.86),
        'Baseline model': (3.35, 4.34),
    }

    x = np.arange(2)  # the label locations
    width = 0.25  # the width of the bars

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, layout='constrained')
    fig.set_figheight(4)
    fig.set_figwidth(10)

    multiplier = 0
    for model, value in values_dice.items():
        offset = width * multiplier
        rects = ax1.bar(x + offset, value, width, label=model)
        ax1.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_hd95.items():
        offset = width * multiplier
        rects = ax2.bar(x + offset, value, width, label="_")
        ax2.bar_label(rects, fmt="{:.1f}", padding=3, fontsize=8.2)
        multiplier += 1

    multiplier = 0
    for model, value in values_MP.items():
        offset = width * multiplier
        rects = ax3.bar(x + offset, value, width, label="_")
        ax3.bar_label(rects, fmt="{:.2f}", padding=3, fontsize=8.2)
        multiplier += 1

    ax1.set_xticks(x + 1*width, scores_dice, fontsize=10)
    ax2.set_xticks(x + 1*width, scores_hd95, fontsize=10)
    ax3.set_xticks(x + 1*width, scores_MP, fontsize=10)
    ax1.set_xlabel('mDice (%)', fontsize=11)
    ax2.set_xlabel('mHD95 (mm)', fontsize=11)
    ax3.set_xlabel('Missed predictions (%)', fontsize=11)
    ax1.set_ylim(70, 85)
    ax2.set_ylim(6.5, 9.5)
    ax3.set_ylim(0, 5)
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])

    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width*1.1, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width*1.1, box2.height*0.9])
    box3 = ax3.get_position()
    ax3.set_position([box3.x0, box3.y0, box3.width*1.1, box3.height*0.9])

    fig.legend(bbox_to_anchor=(0.525,1.0), loc='upper center', ncols=1, fontsize=10)

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

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:purple", "tab:gray"]) 
    # plot_missed_predictions_CT()
    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["plum", "tab:gray"]) 
    # plot_missed_predictions_MRI()

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:red", "tab:blue", "tab:gray"]) 
    # plot_type_results()

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:red", "tab:green", "tab:gray"]) 
    # plot_embeddings_results()

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["tab:red", "tab:orange", "gold", "yellow", "lime", "pink", "tab:gray"]) 
    plot_method_results1()

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["deepskyblue", "tab:blue", "darkblue", "tab:gray"]) 
    # plot_method_results2()

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["mediumpurple", "orchid", "tab:gray"]) 
    # plot_method_results3()

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["darkgreen", "yellow", "tab:gray"]) 
    # plot_method_results4()