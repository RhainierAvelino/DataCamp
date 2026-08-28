"""Search article headlines by meaning instead of matching exact words."""

from openai import OpenAI
from scipy.spatial.distance import cosine


client = OpenAI()
EMBEDDING_MODEL = "text-embedding-3-small"

articles = [
    {"headline": "Economic Growth Continues Amid Global Uncertainty", "topic": "Business", "keywords": ["economy", "business", "finance"]},
    {"headline": "1.5 Billion Tune In to the World Cup Final", "topic": "Sport", "keywords": ["soccer", "World Cup", "television"]},
    {"headline": "New Climate Agreement Sets Global Emissions Targets", "topic": "Environment", "keywords": ["climate", "emissions", "policy"]},
    {"headline": "Breakthrough Treatment Shows Promise in Early Cancer Trial", "topic": "Health", "keywords": ["medicine", "cancer", "research"]},
    {"headline": "Tech Company Unveils Next-Generation Smartphone", "topic": "Technology", "keywords": ["smartphone", "technology", "innovation"]},
    {"headline": "Film Festival Celebrates Independent Directors", "topic": "Culture", "keywords": ["film", "festival", "directors"]},
    {"headline": "Central Bank Holds Interest Rates Steady", "topic": "Finance", "keywords": ["central bank", "interest rates", "economy"]},
    {"headline": "Scientists Discover New Species in the Pacific", "topic": "Science", "keywords": ["discovery", "species", "ocean"]},
    {"headline": "Local Team Wins Championship in Overtime Thriller", "topic": "Sport", "keywords": ["championship", "team", "overtime"]},
    {"headline": "International Summit Focuses on Artificial Intelligence", "topic": "Technology", "keywords": ["artificial intelligence", "summit", "regulation"]},
]


def article_to_text(article):
    """Build one descriptive string from an article dictionary."""
    return (
        f"Headline: {article['headline']}\n"
        f"Topic: {article['topic']}\n"
        f"Keywords: {', '.join(article['keywords'])}"
    )


def create_embeddings(texts):
    """Create embeddings for all strings in ``texts`` in one API request."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def find_closest(query_embedding, embeddings, n=3):
    """Return the indexes and distances of the ``n`` most relevant articles."""
    scores = [
        {"index": index, "distance": cosine(query_embedding, embedding)}
        for index, embedding in enumerate(embeddings)
    ]
    return sorted(scores, key=lambda score: score["distance"])[:n]


article_embeddings = create_embeddings([article_to_text(article) for article in articles])
query = "new developments in artificial intelligence"
query_embedding = create_embeddings([query])[0]

print(f"Search results for: {query}\n")
for hit in find_closest(query_embedding, article_embeddings):
    article = articles[hit["index"]]
    print(f"- {article['headline']} ({article['topic']}; distance: {hit['distance']:.3f})")
