import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class EmbeddingTrainer:
    """
    This is a trainer dedicated for embedding training (e.g. two tower model), and the
    output embedding would be exported and saved for downstream applications
    """

    def __init__(self, model, train_dataset, eval_dataset):
        self.model = model
        self.train_dataloader = DataLoader(train_dataset, batch_size=8192, shuffle=True)
        self.eval_dataloader = DataLoader(eval_dataset, batch_size=1024)

        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        # loss function, in movie len, we treat as a regression task to minimize the rating
        # in the future, we could consider contrastive loss
        self.loss = nn.MSELoss()

        # optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

    def train(self, num_epochs: int = 5):
        print("Start embedding training...")

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for i, (user_ids, item_ids, ratings) in enumerate(self.train_dataloader):
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                self.optimizer.zero_grad()

                predictions = self.model(user_ids, item_ids).squeeze(1)  # B x 1 -> B,
                loss = self.loss(predictions, ratings)

                # backward pass
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % 100 == 99:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.train_dataloader)}], Loss: {running_loss / 100:.4f}")
                    running_loss = 0.0

    def eval(self):
        """
        Eval on the eval dataset and return the loss
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for user_ids, item_ids, ratings in self.eval_dataloader:
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                ratings = ratings.to(self.device)

                predictions = self.model(user_ids, item_ids).squeeze(1)
                loss = self.loss(predictions, ratings)

                total_loss += loss.item()

        print(f"Total evaluation loss: {total_loss:.4f}")
        return total_loss

    def export(self):
        """
        Extract the embeddings of item tower, by invoking the forward
        pass of item tower on all item ids. This is how two tower model
        accelerate inference in production by precompute and caching the
        embedding result
        """

        self.model.eval()

        with torch.no_grad():
            item_ids = torch.LongTensor([i for i in range(self.model.item_cardinality)])
            item_ids = item_ids.to(self.device)
            item_embeddings = self.model.item_embedding(item_ids)
            item_embeddings = self.model.item_tower(item_embeddings)  # num_item x item_dim

        return item_embeddings.cpu().numpy()
