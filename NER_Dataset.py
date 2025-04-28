import torch
from torch.utils.data import Dataset


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
