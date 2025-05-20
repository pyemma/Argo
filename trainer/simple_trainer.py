import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import precision_recall_curve, auc


class SimpleTrainer:
    """
    A simple trainer, with minimal training loop logic
    """

    def __init__(self, model, train_dataset, eval_dataset):
        self.model = model
        self.train_data_loader = DataLoader(dataset=train_dataset, batch_size=32)
        self.eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=32)

        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.model.to(self.device)

        # loss function, for now we use the default BCE
        self.criterion = nn.BCELoss()
        # optimizer, for now we use the default AdamW
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

    def train(self, num_epochs: int = 10):
        print("Simple trainer start to train")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for i, (inputs, labels) in enumerate(self.train_data_loader):
                # move data to the target device
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward pass
                outputs = self.model(inputs).squeeze(1)  # B x 1 shape, remove last dimension
                loss = self.criterion(outputs, labels)

                # backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.train_data_loader)}], Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0

            self.eval()

        print("Simple trainer finished training")

    def eval(self):
        """
        A simple eval function, only support the AUCPR as of now
        """
        print("Start evaluation")

        self.model.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():  # disable gradient computation
            for inputs, labels in self.eval_data_loader:
                inputs = {key: value.to(self.device) for key, value in inputs.items()}

                # in our DIN model, we have applied sigmoid as the final layer to
                # covert logits to probs, thus no additional convert is required here
                outputs = self.model(inputs)

                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

            all_predictions = np.array(all_predictions)
            all_targets = np.array(all_targets)

            precision, recall, _ = precision_recall_curve(all_targets, all_predictions)
            aucpr = auc(recall, precision)
            print(f"Aucpr: {aucpr:.4f}")

    def save(self, model_name, movie_index, path="/tmp"):
        """
        A simple function to save pytorch model
        """
        if not os.path.exists(path):
            print(f"Unable to find path {path}, creating it...")
            os.makedirs(path)
        filename = os.path.join(path, model_name)
        file_path = Path(filename)
        if not file_path.suffix:
            filename += ".pth"
        elif file_path.suffix != ".pth":
            raise ValueError(f"Invalid extension provided {file_path.suffix}")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
        }
        if movie_index:
            checkpoint["movie_index"] = movie_index

        torch.save(checkpoint, filename)
