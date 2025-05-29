import torch
import torch.nn as nn


class TwoTowerModel(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: move to model config
        # here we use user/item to different the 2 towers, which is a common convention in industry
        self.embedding_dim = 32
        self.user_cardinality = 300000
        self.item_cardinality = 350000

        # create embedding lookup table
        self.user_embedding = nn.Embedding(num_embeddings=self.user_cardinality, embedding_dim=self.embedding_dim)
        self.item_embedding = nn.Embedding(num_embeddings=self.item_cardinality, embedding_dim=self.embedding_dim)

        # local tower, linear projection and non-linear activation
        self.user_tower = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),  # this is the actual item embedding dimension that we are outputting
        )

        # merge net, combine user and item tower output for final predictions
        self.merge_net = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, user_ids, item_ids):
        """
        user_ids, B x 1, LongTensor
        item_ids, B x 1, LongTensor
        """
        user_embeddings = self.user_embedding(user_ids)  # B x dim
        item_embeddings = self.item_embedding(item_ids)  # B x dim

        user_hidden = self.user_tower(user_embeddings)  # B x hidden_dim
        item_hidden = self.item_tower(item_embeddings)  # B x hidden_dim

        hidden = torch.concat((user_hidden, item_hidden), dim=-1)  # B x 2 * hidden_dim
        out = self.merge_net(hidden)  # B x 1
        return out


if __name__ == "__main__":
    model = TwoTowerModel()

    user_ids = torch.LongTensor([1, 2, 3]).unsqueeze(1)
    item_ids = torch.LongTensor([1, 2, 3]).unsqueeze(1)

    out = model(user_ids, item_ids)
    print(out)
