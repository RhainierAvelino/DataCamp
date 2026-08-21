import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

client = chromadb.PersistentClient(path="./path/to/save/database")
EMBEDDING_MODEL = "text-embedding-3-small"

# Create a new collection in the database with an embedding function that uses the OpenAI API.
collection = client.create_collection(
    name="netflix_titles", #use as a reference to the collection
    # OpenAIEmbeddingFunction wraps the OpenAI API to generate embeddings for documents.
    embedding_function=OpenAIEmbeddingFunction(
        api_key="YOUR_OPENAI_API_KEY",
        model_name=EMBEDDING_MODEL
    )
)

# Adding embedded data to the collection
result = collection.query( # query the collection for similar documents
    query_texts=["A story about a young wizard who discovers his magical heritage."],
    n_results=3
)

print(result)  # Returns the top 3 most similar documents to the query text