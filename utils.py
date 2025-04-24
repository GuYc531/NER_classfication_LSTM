import numpy as np
from datasets import Dataset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns

def get_max_length_sequence(dataset) -> int:
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

def tokenize_and_align_labels(example, tokenizer, label_map, max_length):
    tokenized_inputs = tokenizer(
        example,                  # a list of tokens (not raw string)
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    word_ids = tokenized_inputs.word_ids()  # aligns subword to word index
    aligned_labels = []

    previous_word_idx = None
    for word_idx in word_ids:
        if word_idx is None:
            aligned_labels.append(-100)  # ignore [CLS], [SEP], [PAD]
        elif word_idx != previous_word_idx:
            aligned_labels.append(label_map[example["ner_tags"][word_idx]])  # first subword
        else:
            # for subword: use same label or "I" tag if you want fine-grain
            label_str = example["ner_tags"][word_idx]
            if label_str.startswith("B-"):
                label_str = label_str.replace("B-", "I-")
            aligned_labels.append(label_map.get(label_str, 0))
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def plot_losses(num_epochs, train_losses, val_losses, val_accuracies) -> None:
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

def plot_confusion_matrix(all_labels,all_preds ) -> None:
    labels_set = sorted(set(all_labels))  # or provide your own label list

    cm = confusion_matrix(all_labels, all_preds, labels=labels_set)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_set, yticklabels=labels_set)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")  # Save locally
    plt.show()