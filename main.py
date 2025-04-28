import torch
import torch.nn as nn
from transformers import AutoTokenizer
from DataPreProcessing import DataHandler
from Train_LSTM_model import Training_lstm
import warnings

warnings.filterwarnings("ignore")

padded_index = -100
tags_column, tokens_column, tokenized_tokens_column = 'tags', 'tokens', 'tokens_tokenized'

data_handler = DataHandler(padded_index=padded_index,
                           tags_column=tags_column,
                           tokens_column=tokens_column,
                           tokenized_tokens_column=tokenized_tokens_column)

data_handler.create_data_frames()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_handler.tokenize_sentences_in_data_frames(tokenizer)

data_handler.create_data_loaders(batch_size=4)

loss_fn = nn.CrossEntropyLoss(ignore_index=padded_index)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_tags = len(data_handler.label_list.values())
best_val_loss = float('inf')
checkpoint_path = "models/"
num_epochs = 1

training = Training_lstm(tokenizer=tokenizer, max_length_sequence=data_handler.max_length_sequence,
                         num_tags=num_tags, device=device)

training.train(data_handler=data_handler, num_epochs=num_epochs, checkpoint_path=checkpoint_path,
               best_val_loss=best_val_loss, loss_fn=loss_fn)

training.plot_results()

training.test_over_test_data_set(data_handler=data_handler, plot_results=True)














