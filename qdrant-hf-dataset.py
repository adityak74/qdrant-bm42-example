from datasets import load_dataset

dataset = load_dataset(
    "Qdrant/arxiv-titles-instructorxl-embeddings", split="train", streaming=True
)

from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")

client.create_collection(
    collection_name="arxiv-titles-instructorxl-embeddings",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
    ),
)

from itertools import islice

def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch

batch_size = 100

for batch in batched(dataset, batch_size):
    ids = [point.pop("id") for point in batch]
    vectors = [point.pop("vector") for point in batch]

    client.upsert(
        collection_name="arxiv-titles-instructorxl-embeddings",
        points=models.Batch(
            ids=ids,
            vectors=vectors,
            payloads=batch,
        ),
    )