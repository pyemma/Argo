import base64
import numpy as np

from datasets import load_dataset

def decode_embedding(base64_string: str) -> np.ndarray:
    """Decodes a Base64 string into a 512-dimensional numpy vector."""
    # Decode from Base64, interpret as a buffer of float32, and reshape.
    return np.frombuffer(
        base64.b64decode(base64_string),
        dtype=np.float32
    ).reshape(-1)

# 1. Load the dataset from the Hugging Face Hub
# Only download the first 1000 samples as the entire dataset is over 1.4 TB
split_name = "train"
num_samples_to_take = 1000
dataset = load_dataset("AL-GR/Item-EMB", split=split_name, streaming=True)

partial_dataset = dataset.take(num_samples_to_take)

# 2. Get a sample from the dataset
it = iter(partial_dataset)
sample = next(it)
item_id = sample['base62_string']
encoded_feature = sample['feature']

print(f"Item ID: {item_id}")
print(f"Encoded Feature (first 50 chars): {encoded_feature[:50]}...")

# 3. Decode the feature string into a vector
embedding_vector = decode_embedding(encoded_feature)
print(embedding_vector.shape)