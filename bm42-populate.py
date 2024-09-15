from qdrant_client import QdrantClient, models
from datasets import load_dataset
import numpy as np

# Initialize the Qdrant client
client = QdrantClient("http://localhost:6333")

# Load the dataset in streaming mode
dataset = load_dataset(
    "Qdrant/arxiv-titles-instructorxl-embeddings", split="train", streaming=True
)

# Iterate over the dataset and insert points into the collection
for i, record in enumerate(dataset):
    # Extract the dense embedding
    print("Processing record {}".format(i))
    dense_vector = {"jina": record['vector']}  # Assuming 'embedding' is the key for dense vector

    # Construct the payload with metadata
    payload = {
        "arxiv_id": record["id"],  # Replace with actual metadata fields
        "title": record["title"],
        "doi": record["DOI"],
    }

    # Insert the record into Qdrant
    client.upsert(
        collection_name="my-hybrid-collection",
        points=[
            models.PointStruct(
                id=i,  # Unique ID for the point, you can also use record ID if available
                vector=dense_vector,
                payload=payload
            )
        ]
    )
