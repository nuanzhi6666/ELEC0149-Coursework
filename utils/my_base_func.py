from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

def PlotConfusionMatrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)

    classes = ["Sell", "Hold", "Buy"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(title)
    plt.colorbar()

    plt.xticks(np.arange(3), classes)
    plt.yticks(np.arange(3), classes)

    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j],
                     ha="center", va="center")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def PlotCurves(plot_data:dict,xlabel,ylabel,title, img_name="loss_curve.png"):
    
    save_path = os.path.join("figures", img_name)
    plt.figure()
    for k,dt in plot_data.items():
        plt.plot(dt, label=k)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def PlotPCAFeaturesVsTarget(
    df,
    x_cols: list[str],
    y_col: str,
    img_name: str
):
    """
    Plot scatter plots of multiple PCA features against target
    in a single figure with subplots.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    x_cols : list[str]
        List of feature column names to plot on x-axis.
    y_col : str
        Target column name.
    img_name : str
        Image file name (saved under figures/).
    """

    os.makedirs("figures", exist_ok=True)

    n = len(x_cols)

    if n == 0:
        return

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(6, 3 * n),
        sharey=True
    )

    # when n == 1, axes is not a list
    if n == 1:
        axes = [axes]

    for i, col in enumerate(x_cols):

        axes[i].scatter(
            df[col],
            df[y_col],
            alpha=0.6,
            s=10
        )

        axes[i].set_xlabel(col)
        axes[i].set_ylabel(y_col)
        axes[i].set_title(f"{col} vs {y_col}")

    plt.tight_layout()

    save_path = os.path.join("figures", img_name)
    plt.savefig(save_path, dpi=150)
    plt.close()


def WriteLog(logfile: str, message: str) -> None:
    """
    Write simple log messages to a local text file.
    Used to track the data acquisition process and possible errors.
    """
    print(message)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logfile, "a") as f:
        f.write(f"{timestamp} - {message}\n")