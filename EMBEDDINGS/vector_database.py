import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

client = chromadb.PersistentClient(path="./path/to/save/database")
EMBEDDING_MODEL = "text-embedding-3-small"


collection = client.create_collection(
    name="my_collection", #use as a reference to the collection
    embedding_function=chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
        api_key="YOUR_OPENAI_API_KEY",
        model_name=EMBEDDING_MODEL
    )
)

client.list_collections()  # Returns a list of all collections in the database

#Adding embedded data to the collection

#single document
collection.add(
    ids=["doc1"],
    documents=["This is the content of document 1."],
)

#multiple documents
collection.add(
    ids=["doc2", "doc3"],
    documents=["This is the content of document 2.", "This is the content of document 3."],
)

#inspect collections
collection.count()  # Returns the number of documents in the collection
collection.peek(3)  # Returns the first 3 documents in the collection

print(client.list_collections())  # Returns a list of all collections in the database