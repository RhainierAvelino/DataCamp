import os

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from sklearn.manifold import TSNE


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

products = [
    {"short_description": "Wireless headphones with noise cancellation", "category": "Audio"},
    {"short_description": "Portable Bluetooth speaker for outdoor music", "category": "Audio"},
    {"short_description": "Smart watch that tracks fitness and health", "category": "Wearables"},
    {"short_description": "Mechanical keyboard for programming", "category": "Computers"},
    {"short_description": "USB-C dock for connecting laptop accessories", "category": "Computers"},
    {"short_description": "Pressure-sensitive tablet for digital drawing", "category": "Creative Tools"},
]

# Extract a list of product short descriptions from products.
product_descriptions = [product["short_description"] for product in products]

# Create embeddings for each product description
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=product_descriptions,
)
response_dict = response.model_dump()

# Extract the embeddings from response_dict and store in products
for index, product in enumerate(products):
    product["embedding"] = response_dict["data"][index]["embedding"]
    
print(products[0].items())

# Create category and embedding lists using list comprehensions.
categories = [product['category'] for product in products]
embeddings = [product['embedding'] for product in products]

# Reduce the embedding dimensions to two using t-SNE so they can be plotted.
# ``perplexity`` must be smaller than the number of products in the dataset.
tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(np.array(embeddings))

# Create a scatter plot from embeddings_2d
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])

for i, category in enumerate(categories):
    plt.annotate(category, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()