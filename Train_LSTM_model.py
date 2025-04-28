from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from LSTM_model import LSTMNERModel
import torch
import utils


class Training_lstm:
    def __init__(self, tokenizer, max_length_sequence, num_tags, device):
        self.tokenizer = tokenizer
        self.num_tags = num_tags
        self.device = device

        self.model = LSTMNERModel(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=max_length_sequence,
            hidden_dim=max_length_sequence * 2,
            num_tags=self.num_tags,
            padding_idx=0
        ).to(self.device)

        self.train_losses = list()
        self.val_losses = list()
        self.val_accuracies = list()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, data_handler, loss_fn, num_epochs: int, checkpoint_path: str, best_val_loss: float):
        self.num_epochs = num_epochs
        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0

            for batch_inputs, batch_labels in tqdm(data_handler.train_dataloader):
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)

                outputs = self.model(batch_inputs)

                loss = loss_fn(outputs.view(-1, self.num_tags), batch_labels.view(-1))
                total_train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            avg_train_loss = total_train_loss / len(data_handler.train_dataloader)
            self.train_losses.append(avg_train_loss)

            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

            self.model.eval()
            total_val_loss = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for input_ids, labels in data_handler.val_dataloader:
                    batch_inputs, batch_labels = input_ids.to(self.device), labels.to(self.device)

                    outputs = self.model(batch_inputs)

                    val_loss = loss_fn(outputs.view(-1, self.num_tags), batch_labels.view(-1))
                    total_val_loss += val_loss.item()

                    preds = torch.argmax(outputs, dim=-1).view(-1)
                    labels = batch_labels.view(-1)

                    mask = labels != data_handler.padded_index
                    filtered_preds = preds[mask]
                    filtered_labels = labels[mask]

                    all_preds.extend(filtered_preds.cpu().numpy())
                    all_labels.extend(filtered_labels.cpu().numpy())

            avg_val_loss = total_val_loss / len(data_handler.val_dataloader)
            self.val_losses.append(avg_val_loss)

            val_acc = accuracy_score(all_labels, all_preds)
            self.val_accuracies.append(val_acc)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), checkpoint_path + f'best_model_epoch_{epoch + 1}.pt')
            print(
                f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

    def plot_results(self):
        utils.plot_losses(num_epochs=self.num_epochs, val_losses=self.val_losses, train_losses=self.train_losses,
                          val_accuracies=self.val_accuracies)

    def test_over_test_data_set(self, data_handler, plot_results=True):

        self.all_preds_test = []
        self.all_labels_test = []

        with torch.no_grad():
            for batch_inputs, batch_labels in data_handler.test_dataloader:
                batch_inputs, batch_labels = batch_inputs.to(self.device), batch_labels.to(self.device)

                outputs = self.model(batch_inputs)  # [batch, seq_len, num_tags]
                preds = torch.argmax(outputs, dim=-1)

                mask = batch_labels != data_handler.padded_index
                filtered_preds = preds[mask]
                filtered_labels = batch_labels[mask]

                self.all_preds_test.extend(filtered_preds.cpu().numpy())
                self.all_labels_test.extend(filtered_labels.cpu().numpy())

        if plot_results:
            print("Classification Report:")
            print(classification_report(self.all_labels_test, self.all_preds_test, digits=4))
            utils.plot_confusion_matrix(all_labels=self.all_labels_test, all_preds=self.all_preds_test)
