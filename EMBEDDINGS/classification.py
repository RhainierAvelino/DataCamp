import os
from openai import OpenAI
from scipy.spatial import distance

EMBEDDING_MODEL = "text-embedding-3-small"


topics = [
    {
        "label": "Tech",
        "description": "Technology, computers, software, artificial intelligence, and electronic devices.",
    },
    {
        "label": "Science",
        "description": "Scientific discoveries, experiments, research, nature, and the study of the world.",
    },
    {
        "label": "Sport",
        "description": "Sports, athletes, teams, competitions, matches, tournaments, and sporting events.",
    },
    {
        "label": "Business",
        "description": "Companies, finance, markets, the economy, investments, and commercial activity.",
    },
]

# Read the API key from an environment variable instead of writing the secret
# directly in this file.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def create_embeddings(texts):
    """Create one embedding vector for each text in ``texts``.

    ``texts`` is the function argument containing the descriptions or article
    texts that should be converted into numeric vectors.
    """
    # ``embeddings.create`` sends the text values to the selected embedding model.
    # ``model`` chooses the model, and ``input`` supplies the text to transform.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    # The API returns one item for every input text.  Keep only the vector from
    # each item because those vectors are what we compare below.
    return [item.embedding for item in response.data]

# Use only the descriptions as the class examples.  A description gives the
# embedding model the meaning of a category without relying on its short label.
# In this list comprehension, ``topic`` is one dictionary from ``topics`` and
# ``topic['description']`` retrieves that dictionary's description value.
class_descriptions = [topic["description"] for topic in topics]
# Create one vector for each class description so article vectors can be compared
# with them in the same embedding space.
class_embeddings = create_embeddings(class_descriptions)

article = {"headline": "How NVIDIA GPUs Could Decide Who Wins the AI Race",
"keywords": ["ai", "business", "computers"]}

def create_article_text(article):
    """Combine one article's fields into one descriptive embedding input.

    ``article`` is the function argument: it is the dictionary supplied by the
    caller.  The returned string gives the embedding model all relevant details.
    """
    # ``join`` combines the keyword list into readable text separated by commas.
    return (
        f"Headline: {article['headline']}\n"
        f"Keywords: {', '.join(article['keywords'])}"
    )
    
article_text = create_article_text(article)
article_embedding = create_embeddings([article_text])[0]

def find_closest(query_vector, embeddings):
    """Find the class vector most similar to the article vector.

    ``query_vector`` is the article embedding, while ``embeddings`` contains one
    vector for each topic description.  The returned index identifies the topic.
    """
    # Store each distance with its index so we can map the closest vector back to
    # the matching topic dictionary later.
    distances = []
    for index, embedding in enumerate(embeddings):
        # Cosine distance compares vector direction.  A smaller value means the
        # article and topic description have more similar meaning.
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    # Sort by distance and return the first result, which is the closest topic.
    return sorted(distances, key=lambda x: x["distance"])[0]

closest = find_closest(article_embedding, class_embeddings)

# Use the matching index to retrieve the human-readable label for the result.
label = topics[closest["index"]]["label"]
print(f"Predicted topic: {label}")
