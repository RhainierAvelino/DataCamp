"""Create a persistent ChromaDB collection and inspect its contents."""

import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


EMBEDDING_MODEL = "text-embedding-3-small"

# Chroma saves this local database folder between program runs.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name=EMBEDDING_MODEL,
)

# get_or_create_collection prevents an error if the collection already exists.
collection = chroma_client.get_or_create_collection(
    name="study_documents",
    embedding_function=embedding_function,
)

# upsert is safe to rerun: an existing ID is updated instead of duplicated.
collection.upsert(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Embeddings represent text as numeric vectors.",
        "A vector database can retrieve documents with similar meaning.",
        "Cosine distance is one way to compare embedding vectors.",
    ],
)

print(f"Number of stored documents: {collection.count()}")
print("\nFirst stored documents:")
for document in collection.peek(limit=3)["documents"]:
    print(f"- {document}")

print("\nCollections:")
for stored_collection in chroma_client.list_collections():
    print(f"- {stored_collection.name}")
