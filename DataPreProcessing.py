from datasets import load_dataset
import utils
import pandas as pd
from NER_Dataset import NERDataset
from torch.utils.data import DataLoader


class DataHandler:
    def __init__(self, padded_index, tags_column,
                 tokens_column,
                 tokenized_tokens_column):
        self.dataset = load_dataset("tner/conll2003")
        # label list taken from https://huggingface.co/datasets/tner/conll2003
        self.label_list = {"O": 0,
                           "B-ORG": 1,
                           "B-MISC": 2,
                           "B-PER": 3,
                           "I-PER": 4,
                           "B-LOC": 5,
                           "I-ORG": 6,
                           "I-MISC": 7,
                           "I-LOC": 8
                           }
        self.max_length_sequence = utils.get_max_length_sequence(self.dataset)
        self.train_data = self.dataset['train']
        self.val_data = self.dataset['validation']
        self.test_data = self.dataset['test']
        self.padded_index = padded_index
        self.tags_column = tags_column
        self.tokens_column = tokens_column
        self.tokenized_tokens_column = tokenized_tokens_column
        self.train_df = pd.DataFrame()
        self.val_df = pd.DataFrame()
        self.test_df = pd.DataFrame()

    def create_data_frames(self):
        """
            creates and processes train, validation, and test into seperate DataFrames.

            returns:None

             the resulting DataFrames are stored as class attributes for further processing.
        """

        self.train_df[self.tokens_column] = [" ".join(i) for i in self.train_data[self.tokens_column]]
        self.val_df[self.tokens_column] = [" ".join(i) for i in self.val_data[self.tokens_column]]
        self.test_df[self.tokens_column] = [" ".join(i) for i in self.test_data[self.tokens_column]]

        self.train_df[self.tags_column] = [
            i + [self.padded_index] * max(0, self.max_length_sequence - len(i)) if len(
                i) < self.max_length_sequence else i for i in
            self.train_data[self.tags_column]]
        self.val_df[self.tags_column] = [
            i + [self.padded_index] * max(0, self.max_length_sequence - len(i)) if len(
                i) < self.max_length_sequence else i
            for i in
            self.val_data[self.tags_column]]
        self.test_df[self.tags_column] = [
            i + [self.padded_index] * max(0, self.max_length_sequence - len(i)) if len(
                i) < self.max_length_sequence else i
            for i in
            self.test_data[self.tags_column]]

    def tokenize_sentences_in_data_frames(self, tokenizer):
        """
            applies a tokenizer to the token sequences in the train, validation, and test DataFrames
            specific on tokenized_tokens_column column.

            in addition, validates that tokenizer returns token's length as expected
            in function _validate_tokens_length().

            params:tokenizer: A tokenizer instance from Hugging Face Transformers used to tokenize the input strings.

            returns: None
        """
        self.train_df[self.tokenized_tokens_column] = self.train_df[self.tokens_column].apply(
            lambda x:
            tokenizer(x, padding='max_length', padding_side='right',
                      max_length=self.max_length_sequence,
                      return_tensors="pt")['input_ids'])
        self.val_df[self.tokenized_tokens_column] = self.val_df[self.tokens_column].apply(
            lambda x: tokenizer(x, padding='max_length', padding_side='right',
                                max_length=self.max_length_sequence,
                                return_tensors="pt")['input_ids'])
        self.test_df[self.tokenized_tokens_column] = self.test_df[self.tokens_column].apply(
            lambda x: tokenizer(x, padding='max_length', padding_side='right',
                                max_length=self.max_length_sequence,
                                return_tensors="pt")['input_ids'])

        self._validate_tokens_length()

    def _validate_tokens_length(self):
        self.train_df = utils.validate_tokenizer_length(self.train_df, self.tokenized_tokens_column,
                                                        self.max_length_sequence)
        self.val_df = utils.validate_tokenizer_length(self.val_df, self.tokenized_tokens_column,
                                                      self.max_length_sequence)
        self.test_df = utils.validate_tokenizer_length(self.test_df, self.tokenized_tokens_column,
                                                       self.max_length_sequence)

    def create_data_loaders(self, batch_size:int=4):
        """
        creates custom dataloader for each train, val, test parts of data set
        :return: None
        """
        train_dataset = NERDataset(dataframe=self.train_df, input_ids_column=self.tokenized_tokens_column,
                                   labels_column=self.tags_column)
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = NERDataset(dataframe=self.val_df, input_ids_column=self.tokenized_tokens_column,
                                 labels_column=self.tags_column)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = NERDataset(dataframe=self.test_df, input_ids_column=self.tokenized_tokens_column,
                                  labels_column=self.tags_column)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
