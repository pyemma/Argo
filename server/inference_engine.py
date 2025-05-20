import traceback

import torch
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel

from data.dataset import DatasetType
from server.model_manager import model_manager, ModelType
from server.retrieval_engine import RetrievalEngine
from server.feature_server import FeatureServer, FeatureName, FeatureServerConfig


# define user request, for now we just use a simple user id
class RecommendationRequest(BaseModel):
    user_id: int


app = FastAPI()

# for now we hard code the model to be used for inference

# initialize candidate generation and feature store
retrieval_engine = RetrievalEngine()
feature_server = FeatureServer(
    FeatureServerConfig(movie_len_history_seq_length=15, movie_len_dataset_type=DatasetType.MOVIE_LENS_LATEST_SMALL, embedding_store_path="artifacts/movie_embeddings.pt")
)


@app.post("/recommend/")
async def recommend(request: RecommendationRequest):
    try:
        # fetch candidates
        candidates = retrieval_engine.generate_candidates()
        user_id = request.user_id

        model = model_manager.get_model(ModelType.DIN_SMALL)
        # there is one implementation issue in the current DIN that the device need to be passed into the module
        # to make it work correct, due to the reason that we are generating a mask within the DIN module right now
        model = model.to("cpu")

        # fetch features
        user_feature = feature_server.extract_user_feature(user_id)
        user_sequence_feature = user_feature[FeatureName.ITEM_SEQUENCE][:8]  # in model training we use sequence length of 8

        # model inference
        # we use a simple approach to predict all candidates within a single batch, there could be
        # OOM issue, but in the current movie len small dataset, this should not cause issue on modern laptop
        num_candidates = len(candidates)
        data = {
            "user_id": torch.tensor([[user_id]]).expand(num_candidates, -1),  # B x 1, use expand is memory efficient
            "item_id": torch.tensor(candidates).unsqueeze(1),  # B x 1
            "user_history_behavior": torch.tensor(user_sequence_feature).expand(num_candidates, -1),  # B x seq_len
            "user_history_length": torch.tensor([8]).expand(num_candidates, -1),  # B x 1
            "dense_features": torch.ones([8]).expand(num_candidates, -1),  # B x num_dense
        }
        with torch.no_grad():
            prediction = model(data)  # B x 1

        sorted_scores, sorted_indices = torch.sort(prediction.squeeze(1), descending=True)
        top_k_scores = sorted_scores[:10].tolist()
        top_k_indices = sorted_indices[:10].tolist()

        # return topk
        return {"recommendations": [{"movie_id": idx, "scores": score, "rank": i} for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores))]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e) + traceback.format_exc())


@app.post("/recommend_transact/")
async def recommend_transact(request: RecommendationRequest):
    # fetch candidates
    candidates = retrieval_engine.generate_candidates()
    user_id = request.user_id

    model = model_manager.get_model(ModelType.TRANSACT_FULL)
    device = model_manager.device

    # fetch features
    user_feature = feature_server.extract_user_feature(user_id)

    # model inference
    # this step is slightly different in training, in the sense that user side feature
    # is shared across all candidate and this might offer an opportunity for us to
    # reduce the memory usage
    batch_size = 256
    num_batch = len(candidates) // batch_size
    predictions = []
    # for simplicity, this is done in a for loop
    # TODO: optimize with batch mode
    for i in range(num_batch):
        candidate_genres, candidate_embeddings = [], []
        start = i * batch_size
        end = min((i + 1) * batch_size, len(candidates))
        for c in candidates[start:end]:
            candidate_feature = feature_server.extract_item_feature(item_id=c)
            candidate_genres.append(candidate_feature[FeatureName.MOVIE_GENRES])
            candidate_embeddings.append(candidate_feature[FeatureName.ITEM_EMBEDDING])

        data = {
            "user_id": torch.tensor([user_id], dtype=torch.int).expand(batch_size),  # B, use expand is memory efficient
            "user_num_viewed_movies": torch.tensor([user_feature[FeatureName.NUM_USER_VIEWED_MOVIES]]).unsqueeze(-1).expand(batch_size, -1),  # B,
            "candidate_genres": torch.tensor(np.array(candidate_genres), dtype=torch.int),  # B x num_genres
            "user_viewed_genres": torch.tensor(user_feature[FeatureName.USER_VIEWED_MOVIE_GENRES], dtype=torch.int).expand(batch_size, -1),  # B x num_genres
            "action_sequence": torch.tensor(user_feature[FeatureName.ACTION_SEQUENCE], dtype=torch.int).expand(batch_size, -1),  # B x seq
            "item_sequence": torch.tensor(user_feature[FeatureName.ITEM_SEQUENCE_EMBEDDING]).expand(batch_size, -1, -1),  # B x seq x d_item
            "candidate": torch.tensor(np.array(candidate_embeddings)),  # B x d_item
        }
        data = {k: v.to(device) for k, v in data.items()}

        with torch.no_grad():
            predictions.append(model(data).to("cpu"))  # B x 1

    predictions = torch.concat(predictions, dim=0).squeeze(1)
    sorted_scores, sorted_indices = torch.sort(predictions, descending=True)
    top_k_scores = sorted_scores[:10].tolist()
    top_k_indices = sorted_indices[:10].tolist()

    # return topk
    return {"recommendations": [{"movie_id": idx, "scores": score, "rank": i} for i, (idx, score) in enumerate(zip(top_k_indices, top_k_scores))]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
