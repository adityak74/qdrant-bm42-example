from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")

from fastembed import SparseTextEmbedding, TextEmbedding

query_text = "Dynamics"

model_bm42 = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
model_jina = TextEmbedding(model_name="jinaai/jina-embeddings-v2-base-en")

sparse_embedding = list(model_bm42.query_embed(query_text))[0]
dense_embedding = list(model_jina.query_embed(query_text))[0]

results = client.query_points(
  collection_name="my-hybrid-collection",
  prefetch=[
      models.Prefetch(query=sparse_embedding.as_object(), using="bm42", limit=100),
      models.Prefetch(query=dense_embedding.tolist(),  using="jina", limit=100),
  ],
  query=models.FusionQuery(fusion=models.Fusion.RRF), # <--- Combine the scores
  limit=10
)

print("Query:", query_text)
print("Search results:")
for result in results.points:
    print("Point {}, Data {}, Score: {}".format(result.id, result.payload["title"], result.score))
    print("---------")
