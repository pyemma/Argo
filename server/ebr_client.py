import grpc

import proto.ebr_pb2
import proto.ebr_pb2_grpc

from configs.server import DEFAULT_EBR_SERVER_PORT

"""
    A dummy example to send request to the ebr server to retrieve the candidates
"""

class EBRClient:

    def __init__(self):
        channel = grpc.insecure_channel(f"localhost:{DEFAULT_EBR_SERVER_PORT}")
        self.stub = proto.ebr_pb2_grpc.EBRServiceStub(channel)
    

    def send_request(self, query, num_candidate):
        search_response = self.stub.Search(proto.ebr_pb2.SearchRequest(
            ids=query,
            k=num_candidate
        ))

        print("Search results:")
        for result in search_response.results:
            print(f"ID: {result.id}, Score: {result.score}, Source ID: {result.source_id}")
        return search_response.results

if __name__ == "__main__":
    client = EBRClient()
    client.send_request([1, 2, 3], 3)