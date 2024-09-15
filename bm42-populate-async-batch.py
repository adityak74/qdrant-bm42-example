import asyncio
from itertools import islice
from datasets import load_dataset
from tqdm.asyncio import tqdm
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding  # Adjust import according to your setup

# Initialize your models
model_bm42 = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
model_jina = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")

# Compute embeddings
def compute_embeddings(texts):
    sparse_embeddings = [list(model_bm42.query_embed(text))[0] for text in texts]
    dense_embeddings = [list(model_jina.query_embed(text))[0] for text in texts]
    return dense_embeddings, sparse_embeddings

async def upsert_batch(client, collection_name, batch):
    ids = [point["id"] for point in batch]
    titles = [point["title"] for point in batch]

    # Compute embeddings for the batch
    dense_vectors, sparse_vectors = compute_embeddings(titles)

    # Convert sparse vectors to the format expected by Qdrant
    sparse_vectors_formatted = [
        models.SparseVector(
            indices=sparse_vector.indices,
            values=sparse_vector.values
        )  # Convert to list of indices and values
        for sparse_vector in sparse_vectors
    ]

    # Upsert the batch into Qdrant
    client.upsert(
        collection_name=collection_name,
        points=models.Batch(
            ids=ids,
            vectors={
                "jina": dense_vectors,
                "bm42": sparse_vectors_formatted
            },
            payloads=batch,
        ),
        wait=False,
    )

async def process_batches(dataset, batch_size, total_items, client, collection_name):
    async for batch in tqdm(batched(dataset, batch_size), total=total_items // batch_size, desc="Uploading Batches"):
        await upsert_batch(client, collection_name, batch)

# Define the batched method
def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch

# Main asynchronous function
async def main():
    # Initialize the Qdrant client
    client = QdrantClient("http://localhost:6333")  # Modify as needed for async
    collection_name = "my-hybrid-collection"

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name="my-hybrid-collection",
        vectors_config={
            "jina": models.VectorParams(
                size=768,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "bm42": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }
    )

    # Load the dataset in streaming mode
    dataset = load_dataset(
        "Qdrant/arxiv-titles-instructorxl-embeddings", split="train", streaming=True
    )

    batch_size = 500  # Adjust the batch size as needed
    total_items = 2_500_000  # Set this based on known dataset size or other criteria

    await process_batches(dataset, batch_size, total_items, client, collection_name)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
