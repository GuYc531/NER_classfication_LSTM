import numpy as np
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns


def get_max_length_sequence(dataset) -> int:
    """

    :param dataset: dataset of choice  Dataset object
    :return: computes the max length sequence across all parts of data set,
     and return the closets number of power of 2 higher than max number which found earlier
    """
    max_list = list()
    for part in dataset.keys():
        max_list.append(np.max([len(i) for i in dataset[f'{part}']['tokens']]))

    max_length = get_upper_nearest_power_of_2(max(max_list))

    return max_length


def get_upper_nearest_power_of_2(n: int) -> int:
    '''
    This function finds the upper nearest power of 2 to a given number 'n'.
    If the number is a power of two it returns the same number, otherwise, it
    computes the powers less than and more than 'n' and returns the nearest one.
    '''

    # Handling 0 and negative numbers exclusively
    if n <= 0:
        raise ValueError('Number must be positive')

    i = 2
    while True:
        current_power = 2 ** i
        if current_power < n:
            i += 1
            continue
        else:
            return current_power


def validate_tokenizer_length(df, col, max_len):
    """

    :param df: pd.DataFrame
    :param col: column name
    :param max_len: the max length of tensor for input for following model
    :return: comutes if a column in df has value which exceeds `max_len` and if it does returns only max_len charactes instead
    not ideal solution but due to time limit i solved it with the easy approach
    """
    if col not in df.columns:
        raise ValueError(f"{col} is not in given data frame columns, must be one of {df.columns}")
    df[col] = df[col].apply(lambda x: x if x.shape[1] == max_len else x[:, :max_len])
    return df

def plot_losses(num_epochs, train_losses, val_losses, val_accuracies) -> None:
    """

    :param num_epochs:
    :param train_losses:
    :param val_losses:
    :param val_accuracies:
    :return: plots the loss of train and validation in subplot 1 and the accuarcy of validion in 2
    """
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()

    plt.savefig("plots/Loss.png")
    plt.close()


def plot_confusion_matrix(all_labels, all_preds) -> None:
    """

    :param all_labels:
    :param all_preds:
    :return: plots confusion matrix of predicted labels vs true labels
    """
    labels_set = sorted(set(all_labels))

    cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_set, yticklabels=labels_set)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix.png")
    plt.show()
