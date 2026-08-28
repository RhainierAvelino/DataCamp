"""Embed several search queries in one request and find the best document for each."""

from openai import OpenAI
from scipy.spatial.distance import cosine


client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

documents = [
    "Wireless headphones with active noise cancellation.",
    "A compact Bluetooth speaker for outdoor music.",
    "A mechanical keyboard designed for programming.",
    "A pressure-sensitive tablet for digital illustration.",
]
queries = ["quiet headphones for a flight", "tools for making digital art"]


def create_embeddings(texts):
    """Create one embedding per string; returned vectors keep the input order."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def best_match(query_embedding, document_embeddings):
    """Return the document index with the smallest cosine distance."""
    return min(
        range(len(document_embeddings)),
        key=lambda index: cosine(query_embedding, document_embeddings[index]),
    )


# Batch requests reduce the number of API calls compared with embedding one item at a time.
document_embeddings = create_embeddings(documents)
query_embeddings = create_embeddings(queries)

for query, query_embedding in zip(queries, query_embeddings):
    match_index = best_match(query_embedding, document_embeddings)
    print(f"Query: {query}")
    print(f"Best match: {documents[match_index]}\n")
