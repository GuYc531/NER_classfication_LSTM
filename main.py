import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
import torch.nn as nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils
import os

import warnings
warnings.filterwarnings("ignore")


class LSTMNERModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        logits = self.fc(lstm_out)  # [batch_size, seq_len, num_tags]
        return logits


class NERDataset(Dataset):
    def __init__(self, dataframe, input_ids_column, labels_column):
        self.df = dataframe
        self.input_ids_column = input_ids_column
        self.labels_column = labels_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        input_ids = torch.tensor(item[self.input_ids_column], dtype=torch.long).squeeze(0)
        labels = torch.tensor(item[self.labels_column], dtype=torch.long).squeeze(0)
        return input_ids, labels


# load dataset
dataset = load_dataset("tner/conll2003")
# label list taken from https://huggingface.co/datasets/tner/conll2003
label_list = {"O": 0,
              "B-ORG": 1,
              "B-MISC": 2,
              "B-PER": 3,
              "I-PER": 4,
              "B-LOC": 5,
              "I-ORG": 6,
              "I-MISC": 7,
              "I-LOC": 8
              }

max_length_sequence = utils.get_max_length_sequence(dataset)

train_data, val_data, test_data = dataset['train'], dataset['validation'], dataset['test']

train_df, val_df, test_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
tags_column, tokens_column, tokenized_tokens_column = 'tags', 'tokens', 'tokens_tokenized'
padded_index = -100

train_df[tokens_column] = [" ".join(i) for i in train_data[tokens_column]]
val_df[tokens_column] = [" ".join(i) for i in val_data[tokens_column]]
test_df[tokens_column] = [" ".join(i) for i in test_data[tokens_column]]

train_df[tags_column] = [i + [padded_index] * max(0, max_length_sequence - len(i)) if len(i) < max_length_sequence else i for i in
                         train_data[tags_column]]
val_df[tags_column] = [i + [padded_index] * max(0, max_length_sequence - len(i)) if len(i) < max_length_sequence else i for i in
                       val_data[tags_column]]
test_df[tags_column] = [i + [padded_index] * max(0, max_length_sequence - len(i)) if len(i) < max_length_sequence else i for i in
                        test_data[tags_column]]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_df[tokenized_tokens_column] = train_df[tokens_column].apply(lambda x:
                                                                  tokenizer(x, padding='max_length',
                                                                            padding_side='right',
                                                                            max_length=max_length_sequence,
                                                                            return_tensors="pt")[
                                                                      'input_ids'])
val_df[tokenized_tokens_column] = val_df[tokens_column].apply(
    lambda x: tokenizer(x, padding='max_length', padding_side='right',
                        max_length=max_length_sequence,
                        return_tensors="pt")['input_ids'])
test_df[tokenized_tokens_column] = test_df[tokens_column].apply(
    lambda x: tokenizer(x, padding='max_length', padding_side='right',
                        max_length=max_length_sequence,
                        return_tensors="pt")['input_ids'])

train_dataset = NERDataset(dataframe=train_df, input_ids_column=tokenized_tokens_column, labels_column=tags_column)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # , collate_fn=collate_fn)

val_dataset = NERDataset(dataframe=val_df, input_ids_column=tokenized_tokens_column, labels_column=tags_column)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)  # , collate_fn=collate_fn)

test_dataset = NERDataset(dataframe=test_df, input_ids_column=tokenized_tokens_column, labels_column=tags_column)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)  # , collate_fn=collate_fn)

loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # or your pad tag ID
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_tags = len(label_list.values())
best_val_loss = float('inf')
checkpoint_path = "models/"  # or include directory like "checkpoints/best_model.pt"

model = LSTMNERModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=max_length_sequence,
    hidden_dim=max_length_sequence * 2,
    num_tags=num_tags,
    padding_idx=0
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1

train_losses, val_losses, val_accuracies = list(), list(), list()

for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0

    for batch_inputs, batch_labels in tqdm(train_dataloader):
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        outputs = model(batch_inputs)


        loss = loss_fn(outputs.view(-1, num_tags), batch_labels.view(-1))
        total_train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if total_train_loss > 50:
            break

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for input_ids, labels in val_dataloader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

            outputs = model(batch_inputs)  # [batch, seq_len, num_tags]

            # reshape for loss: [batch*seq_len, num_tags], [batch*seq_len]
            val_loss = loss_fn(outputs.view(-1, num_tags), batch_labels.view(-1))
            total_val_loss += val_loss.item()

            preds = torch.argmax(outputs, dim=-1).view(-1)
            labels = batch_labels.view(-1)

            # Only include non-padded labels
            mask = labels != padded_index
            filtered_preds = preds[mask]
            filtered_labels = labels[mask]

            all_preds.extend(filtered_preds.cpu().numpy())
            all_labels.extend(filtered_labels.cpu().numpy())
            if total_train_loss > 50:
                break

    avg_val_loss = total_val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    val_acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_acc)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), checkpoint_path + f'best_model_epoch_{epoch+1}.pt')
    print(
        f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

utils.plot_losses(num_epochs=num_epochs, val_losses=val_losses, train_losses=train_losses,
                  val_accuracies=val_accuracies)

utils.plot_confusion_matrix(all_labels=all_labels, all_preds=all_preds)

all_preds_test = []
all_labels_test = []

with torch.no_grad():
    for batch_inputs, batch_labels in test_dataloader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        outputs = model(batch_inputs)  # [batch, seq_len, num_tags]
        preds = torch.argmax(outputs, dim=-1)

        # Flatten and filter out ignored labels (-100)
        mask = batch_labels != padded_index
        filtered_preds = preds[mask]
        filtered_labels = batch_labels[mask]

        all_preds_test.extend(filtered_preds.cpu().numpy())
        all_labels_test.extend(filtered_labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels_test, all_preds_test, digits=4))
utils.plot_confusion_matrix(all_labels=all_labels_test, all_preds=all_preds_test)

input_ids = tokenizer(["EU rejects German call to boycott British lamb ."],
                      padding='max_length', padding_side='right',
                      max_length=max_length_sequence,
                      return_tensors="pt")['input_ids']

model.eval()
with torch.no_grad():
    logits = model(input_ids.to(device))  # [batch_size, seq_len, num_tags]
predictions = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
print(predictions)
print(0)
