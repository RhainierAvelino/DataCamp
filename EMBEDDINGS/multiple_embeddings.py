"""Create product embeddings in one request and visualise their relationships."""

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.manifold import TSNE


client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

products = [
    {"description": "Wireless headphones with noise cancellation", "category": "Audio"},
    {"description": "Portable Bluetooth speaker for outdoor music", "category": "Audio"},
    {"description": "Smart watch that tracks fitness and health", "category": "Wearables"},
    {"description": "Mechanical keyboard for programming", "category": "Computers"},
    {"description": "USB-C dock for connecting laptop accessories", "category": "Computers"},
    {"description": "Pressure-sensitive tablet for digital drawing", "category": "Creative Tools"},
]

# Sending a list creates one embedding per description, in the same order.
response = client.embeddings.create(
    model=EMBEDDING_MODEL,
    input=[product["description"] for product in products],
)
embeddings = np.array([item.embedding for item in response.data])

# t-SNE reduces high-dimensional embeddings to two dimensions for a plot only.
# Its perplexity must be smaller than the number of data points.
coordinates = TSNE(n_components=2, perplexity=3, random_state=42).fit_transform(embeddings)

plt.figure(figsize=(8, 5))
plt.scatter(coordinates[:, 0], coordinates[:, 1])
for index, product in enumerate(products):
    plt.annotate(product["category"], coordinates[index])

plt.title("Product embeddings visualised with t-SNE")
plt.tight_layout()
plt.show()
