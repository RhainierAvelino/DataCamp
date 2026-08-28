"""Recommend unseen products from a user's viewing history."""

import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine


client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

products = [
    {"name": "Wireless Noise-Cancelling Headphones", "category": "Audio", "description": "Comfortable headphones with clear sound and long battery life.", "keywords": ["music", "wireless", "noise cancelling"]},
    {"name": "Portable Bluetooth Speaker", "category": "Audio", "description": "A compact speaker for powerful sound at home or outdoors.", "keywords": ["music", "portable", "bluetooth"]},
    {"name": "Smart Fitness Watch", "category": "Wearables", "description": "Tracks workouts, heart rate, sleep, and daily activity.", "keywords": ["fitness", "health", "smartwatch"]},
    {"name": "Ergonomic Office Chair", "category": "Office", "description": "An adjustable chair designed for comfortable desk work.", "keywords": ["office", "desk", "comfort"]},
    {"name": "Mechanical Keyboard", "category": "Computers", "description": "A responsive keyboard for programming and productive work.", "keywords": ["typing", "programming", "computer"]},
    {"name": "USB-C Laptop Dock", "category": "Computers", "description": "Connects a laptop to monitors, storage, and other accessories.", "keywords": ["laptop", "workspace", "USB-C"]},
    {"name": "Digital Drawing Tablet", "category": "Creative Tools", "description": "A pressure-sensitive tablet for digital art and illustration.", "keywords": ["drawing", "design", "creative"]},
]


def product_to_text(product):
    """Combine all useful product fields into the text sent to the embedding model."""
    return (
        f"Name: {product['name']}\n"
        f"Category: {product['category']}\n"
        f"Description: {product['description']}\n"
        f"Keywords: {', '.join(product['keywords'])}"
    )


def create_embeddings(texts):
    """Return one embedding vector for every product text in ``texts``."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def find_closest(query_embedding, candidate_embeddings, n=3):
    """Return up to ``n`` candidate indexes, ordered from most to least similar."""
    scores = [
        {"index": index, "distance": cosine(query_embedding, embedding)}
        for index, embedding in enumerate(candidate_embeddings)
    ]
    return sorted(scores, key=lambda score: score["distance"])[:n]


# These indexes represent products that the user has already viewed or purchased.
history_indexes = [0, 4]
history_products = [products[index] for index in history_indexes]
unseen_products = [product for index, product in enumerate(products) if index not in history_indexes]

# An average embedding is a simple profile of the user's combined interests.
history_embeddings = create_embeddings([product_to_text(product) for product in history_products])
user_profile = np.mean(history_embeddings, axis=0)
unseen_embeddings = create_embeddings([product_to_text(product) for product in unseen_products])

# Recommendations exclude history, so the output never suggests an item already viewed.
profile_hits = find_closest(user_profile, unseen_embeddings)
print("Recommendations based on your history:")
for hit in profile_hits:
    product = unseen_products[hit["index"]]
    print(f"- {product['name']} (distance: {hit['distance']:.3f})")

# This second view finds products similar to only the most recent interaction.
last_product = history_products[-1]
last_embedding = history_embeddings[-1]
similar_hits = find_closest(last_embedding, unseen_embeddings)

print(f"\nBecause you viewed: {last_product['name']}")
for hit in similar_hits:
    product = unseen_products[hit["index"]]
    print(f"- {product['name']} (distance: {hit['distance']:.3f})")
