from itertools import islice
from qdrant_client import QdrantClient, models
from datasets import load_dataset
from tqdm import tqdm

# Initialize the Qdrant client
client = QdrantClient("http://localhost:6333")

# Load the dataset in streaming mode
dataset = load_dataset(
    "Qdrant/arxiv-titles-instructorxl-embeddings", split="train", streaming=True
)

# Define the batched method
def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch

batch_size = 500  # Adjust the batch size as needed

# Estimate total number of items (optional, if possible)
total_items = 2_500_000  # Set this based on known dataset size or other criteria

# Iterate over batches of the dataset and upsert them
for batch in tqdm(batched(dataset, batch_size), total=total_items // batch_size, desc="Uploading Batches"):
    # Extract IDs and vectors from the batch
    ids = [point.pop("id") for point in batch]  # Assuming 'id' is the key for unique IDs
    vectors = [point.pop("vector") for point in batch]  # Assuming 'vector' is the key for dense vectors

    # Upsert the batch into Qdrant
    client.upsert(
        collection_name="my-hybrid-collection",
        points=models.Batch(
            ids=ids,
            vectors={"jina": vectors},
            payloads=batch,  # The remaining data in each point will be used as the payload
        ),
    )

print("Batch upload completed.")
