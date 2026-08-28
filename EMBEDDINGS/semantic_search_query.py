"""Use ChromaDB to store documents and perform a semantic search query."""

import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


EMBEDDING_MODEL = "text-embedding-3-small"

# PersistentClient saves the database in this folder so it is available next time.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name=EMBEDDING_MODEL,
)

# get_or_create_collection makes this sample safe to run more than once.
collection = chroma_client.get_or_create_collection(
    name="netflix_titles",
    embedding_function=embedding_function,
)

documents = [
    "A young wizard discovers his magical heritage and attends a school of magic.",
    "A group of astronauts travel through a wormhole to save humanity.",
    "A detective investigates a mystery in a small coastal town.",
]

# upsert adds new documents or replaces documents with the same IDs.
collection.upsert(
    ids=["wizard", "space", "detective"],
    documents=documents,
)

query = "A fantasy adventure about a student learning magic"
results = collection.query(query_texts=[query], n_results=3)

print(f"Search results for: {query}\n")
for document, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"- {document} (distance: {distance:.3f})")
