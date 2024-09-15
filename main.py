# Import client library
from qdrant_client import QdrantClient
from tqdm import tqdm

client = QdrantClient(url="http://localhost:6333")

client.set_model("sentence-transformers/all-MiniLM-L6-v2")
# comment this line to use dense vectors only
client.set_sparse_model("prithivida/Splade_PP_en_v1")

if not client.collection_exists("startups"):
    client.create_collection(
        collection_name="startups",
        vectors_config=client.get_fastembed_vector_params(),
        # comment this line to use dense vectors only
        sparse_vectors_config=client.get_fastembed_sparse_vector_params(),
    )

import json

payload_path = "startups_demo.json"
metadata = []
documents = []

with open(payload_path) as fd:
    for line in fd:
        obj = json.loads(line)
        documents.append(obj.pop("description"))
        metadata.append(obj)


if __name__ == "__main__":
    client.add(
        collection_name="startups",
        documents=documents,
        metadata=metadata,
        ids=tqdm(range(len(documents))),
        # Requires wrapping code into if __name__ == '__main__' block
    )
