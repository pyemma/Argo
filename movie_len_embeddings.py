import torch

from data.dataset import prepare_movie_len_rating_dataset, DatasetType
from model.two_tower import TwoTowerModel
from trainer.embedding_trainer import EmbeddingTrainer


NUM_TRIAL = 3

if __name__ == "__main__":
    train_dataset, eval_dataset = prepare_movie_len_rating_dataset(eval_ratio=0.05, dataset_type=DatasetType.MOVIE_LENS_LATEST_FULL)
    best_embeddings, best_embeddings_loss = None, None

    # we train and eval embedding for serval trails to find the best parameters
    for trial in range(NUM_TRIAL):
        model = TwoTowerModel()
        trainer = EmbeddingTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
        trainer.train(num_epochs=1)
        loss = trainer.eval()

        if best_embeddings_loss is None or loss < best_embeddings_loss:
            best_embeddings_loss = loss
            best_embeddings = trainer.export()
            print(f"On trial {trial}, found the best loss {loss:.4f}, current best {best_embeddings_loss:.4f}...")

    torch.save(best_embeddings, "artifacts/movie_embeddings_v2.pt")
