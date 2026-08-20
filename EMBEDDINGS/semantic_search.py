# Import the libraries needed to work with environment variables,
# call the OpenAI API, and compute vector similarity.
import os  # Lets us read configuration values, such as the API key, from the environment.
import openai as OpenAI  # The OpenAI Python package used to request text embeddings.
import numpy as np  # Provides numerical array utilities for working with vectors.
from sklearn.manifold import TSNE  # Can reduce vectors to two dimensions for visualization.
from scipy.spatial import distance  # Provides cosine distance for comparing vectors.

# Initialize the OpenAI client using the API key stored in the environment.
# ``os.environ[...]`` reads the variable and raises an error if it has not been set.
# Keeping the key in an environment variable is safer than writing it directly in code.
client = OpenAI.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Each article is a dictionary with a headline, topic, and keywords.
# These items will later be converted into text and then embeddings.
# A list of dictionaries is convenient here because each article has named fields.
# The keywords give the embedding model additional information about the article.
articles = [
    {"headline": "Economic Growth Continues Amid Global Uncertainty",
     "topic": "Business",
     "keywords": ["economy", "business", "finance"]},
    {"headline": "1.5 Billion Tune In to the World Cup Final",
     "topic": "Sport",
     "keywords": ["soccer", "world cup", "tv"]},
    {"headline": "New Climate Agreement Sets Global Emissions Targets",
     "topic": "Environment",
     "keywords": ["climate", " emissions", "policy"]},
    {"headline": "Breakthrough Treatment Shows Promise in Early Cancer Trial",
     "topic": "Health",
     "keywords": ["medicine", "cancer", "research"]},
    {"headline": "Tech Company Unveils Next-Generation Smartphone",
     "topic": "Technology",
     "keywords": ["smartphone", "technology", "innovation"]},
    {"headline": "Film Festival Celebrates Independent Directors",
     "topic": "Culture",
     "keywords": ["film", "festival", "directors"]},
    {"headline": "Central Bank Holds Interest Rates Steady",
     "topic": "Finance",
     "keywords": ["central bank", "interest rates", "economy"]},
    {"headline": "Scientists Discover New Species in the Pacific",
     "topic": "Science",
     "keywords": ["discovery", "species", "ocean"]},
    {"headline": "Local Team Wins Championship in Overtime Thriller",
     "topic": "Sport",
     "keywords": ["championship", "team", "overtime"]},
    {"headline": "International Summit Focuses on Artificial Intelligence",
     "topic": "Technology",
     "keywords": ["artificial intelligence", "summit", "regulation"]},
]

# Turn each article into a single text block so it can be embedded as a string.
def create_article_text(article):
    # ``article`` is one dictionary from the list above.  Combining its fields into
    # one string gives the embedding API a single, descriptive input value.
    return (
        f"""Headline: {article['headline']}
        Topic: {article['topic']}
        Keywords: {', '.join(article['keywords'])}"""
    )


# Convert each article into a text string before creating embeddings.
# The list comprehension calls the function once for every article.
article_texts = [create_article_text(article) for article in articles]

# Generate vector embeddings for all article texts so semantic similarity
# can be compared in a shared embedding space.  The helper should return one
# numeric vector per input text, in the same order as ``article_texts``.
article_embeddings = create_article_embeddings(article_texts)

# Find the closest articles to a query vector by computing cosine distance.
# Lower distance means the vectors are more similar.
def find_n_closest_articles(query_vector, embeddings, n=3):
    """Return the ``n`` articles whose vectors are closest to a query vector.

    ``query_vector`` is the embedding of the user's search text, while
    ``embeddings`` contains the article vectors.  ``n=3`` is a default argument,
    so callers receive three results unless they request a different number.
    """
    # Store both the calculated distance and the original article index so the
    # winning vector can later be matched back to its article dictionary.
    distance_scores = []
    for i, embedding in enumerate(embeddings):
        # Cosine distance compares the direction of vectors rather than their
        # length.  A smaller value means greater semantic similarity.
        dist = distance.cosine(query_vector, embedding)
        distance_scores.append({"distance": dist, "index": i})
    # Sort from most similar to least similar, then keep only the first ``n``.
    distance_sorted = sorted(distance_scores, key=lambda x: x["distance"])
    return distance_sorted[:n]


# Example search query and its embedding.  The query must be embedded with the
# same model as the articles so both vectors use the same coordinate system.
query_text = "AI"
query_vector = create_article_embeddings([query_text])[0]

# Search for the most relevant articles based on semantic similarity.
hits = find_n_closest_articles(query_vector, article_embeddings)

# Each hit contains an ``index`` created by ``find_n_closest_articles``.  Use it
# to retrieve the complete article, rather than printing only the vector data.
for hit in hits:
    article = articles[hit["index"]]
    print(f"Headline: {article['headline']}")
          


