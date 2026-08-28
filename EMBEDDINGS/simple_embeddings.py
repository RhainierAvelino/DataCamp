"""Create and inspect one text embedding."""

from openai import OpenAI


client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-large"
text = "The quick brown fox jumped over the lazy dog."

# An embedding is a list of numbers that represents the meaning of this text.
response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
embedding = response.data[0].embedding

print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
