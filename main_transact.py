import torch

from configs.model import LARGE_TRANSACT_CONFIG, OUTPUT_MODEL_PATH
from data.dataset import prepare_movie_len_transact_dataset, DatasetType
from model.transact import TransAct
from trainer.simple_trainer import SimpleTrainer


DEFAULT_MOVIE_LEN_EMBEDDING_PATH = "artifacts/movie_embeddings.pt"

if __name__ == "__main__":
    embedding_store = torch.load(DEFAULT_MOVIE_LEN_EMBEDDING_PATH, weights_only=False)

    model = TransAct(LARGE_TRANSACT_CONFIG)
    train_dataset, eval_dataset = prepare_movie_len_transact_dataset(embedding_store=embedding_store, dataset_type=DatasetType.MOVIE_LENS_LATEST_FULL)

    trainer = SimpleTrainer(model=model, train_dataset=train_dataset, eval_dataset=eval_dataset)
    trainer.train(num_epochs=2)

    trainer.save("transact-movie-len-full", movie_index=None, path=OUTPUT_MODEL_PATH)
