"""Classify an article by comparing its embedding with topic descriptions."""

from openai import OpenAI
from scipy.spatial.distance import cosine


# OpenAI() reads the already configured OPENAI_API_KEY environment variable.
client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

# A detailed description gives each label a meaningful vector representation.
topics = [
    {"label": "Technology", "description": "Computers, software, artificial intelligence, and electronic devices."},
    {"label": "Science", "description": "Scientific research, experiments, nature, and discoveries about the world."},
    {"label": "Sport", "description": "Athletes, teams, matches, tournaments, and other sporting events."},
    {"label": "Business", "description": "Companies, markets, finance, investments, and commercial activity."},
]


def create_embeddings(texts):
    """Return one embedding for every string in ``texts``."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def article_to_text(article):
    """Turn the article fields into the one text string that will be embedded."""
    return f"Headline: {article['headline']}\nKeywords: {', '.join(article['keywords'])}"


def find_closest_embedding(query_embedding, embeddings):
    """Return the index and cosine distance of the most similar embedding."""
    scores = [
        {"index": index, "distance": cosine(query_embedding, embedding)}
        for index, embedding in enumerate(embeddings)
    ]
    # Lower cosine distance means the two texts have more similar meaning.
    return min(scores, key=lambda score: score["distance"])


article = {
    "headline": "How NVIDIA GPUs Could Decide Who Wins the AI Race",
    "keywords": ["AI", "business", "computers"],
}

# Embed the class examples and the article with the same model.
topic_embeddings = create_embeddings([topic["description"] for topic in topics])
article_embedding = create_embeddings([article_to_text(article)])[0]

closest_topic = find_closest_embedding(article_embedding, topic_embeddings)
prediction = topics[closest_topic["index"]]["label"]

print(f"Predicted topic: {prediction}")
print(f"Cosine distance: {closest_topic['distance']:.3f}")
