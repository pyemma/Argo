from concurrent import futures
import heapq
import os
import pickle

import faiss
import torch

import grpc

from configs.model import OUTPUT_MODEL_PATH
from configs.server import DEFAULT_EBR_SERVER_PORT
import proto.ebr_pb2
import proto.ebr_pb2_grpc


class EBRServer(proto.ebr_pb2_grpc.EBRServiceServicer):
    """
    This is a simple abstraction around the EBR (embedding based retrieval),
    where we would leverage different types of embedding to preform ANN search.

    This is different from the traditional retrieval such as token-based or
    collaborate-based in that we could use the semantic information encoded within
    the embeddings to identity candidates in a latent learned space, which is not
    directly observable
    """

    def __init__(self):
        self.embedding_store = torch.load(os.path.join(OUTPUT_MODEL_PATH, "movie_embeddings_v2.pt"), weights_only=False)
        self.faiss_index = faiss.read_index(os.path.join(OUTPUT_MODEL_PATH, "movies.index"))
        with open(os.path.join(OUTPUT_MODEL_PATH, "movie_id_remapper.pkl"), "rb") as file:
            self.movie_id_remapper = pickle.load(file)

    def Search(self, request, context):
        query = request.ids
        num_candidates = request.k

        candidates = self.generate_candidates(query=query, num_candidate=num_candidates)
        results = []
        for c in candidates:
            results.append(proto.ebr_pb2.SearchResult(
                id=c["id"], score=c["score"], source_id=c["source_id"]
            ))
        return proto.ebr_pb2.SearchResponse(results=results)

    def generate_candidates(self, query, num_candidate):
        """
        Return the movie id and distance as score for each input query

        The query could be a list of candidate that user have interacted with before, we
        would fetch their embeddings from the embedding store, and then use ANN to retrieve
        the candidates, and then do a k-merge sort to return the candidates based on number
        requested
        """
        print(f"number of candidates within query: {len(query)}")
        # we don't need to do remapping here because the embedding store contains a contiguous
        # id in the range of 0 to maximum movie id in MovieLen dataset
        query_embeddings = self.embedding_store[query]
        distances, indices = self.faiss_index.search(query_embeddings, num_candidate)
        # the distance and indices are sorted in increasing order, we use k-merge sort to
        # return the candidates
        heap, visited = [], set(query)
        for k in range(len(query)):
            heapq.heappush(heap, (distances[k][0], k, 0))  # distance, array_index, index
        result = []
        while len(result) < num_candidate:
            # get the next candidate to check
            dist, array_index, index = heapq.heappop(heap)
            candidate = self.movie_id_remapper[indices[array_index][index]]
            if candidate not in visited:
                visited.add(candidate)
                result.append({"id": candidate, "score": float(dist), "source_id": query[array_index]})
            # push the next candidate from the array into the heap
            next_index = index + 1
            if next_index < len(indices[array_index]):  # check if we have consumed all elements in this array
                heapq.heappush(heap, (distances[array_index][next_index], array_index, next_index))

        return result

def server():
    """
        Launch the service
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    proto.ebr_pb2_grpc.add_EBRServiceServicer_to_server(EBRServer(), server)
    server.add_insecure_port(f'[::]:{DEFAULT_EBR_SERVER_PORT}')
    server.start()
    print(f"Server started on port {DEFAULT_EBR_SERVER_PORT}")
    server.wait_for_termination()

if __name__ == '__main__':
    server()