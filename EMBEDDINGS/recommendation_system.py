# ``os`` lets Python read values from the operating system, including our API key.
import os

# NumPy provides tools for working with numeric arrays, such as averaging vectors.
import numpy as np
# ``OpenAI`` is the client class used to send requests to the OpenAI API.
from openai import OpenAI
# SciPy provides the cosine-distance function used to compare two vectors.
from scipy.spatial import distance


# Create an API client using the key stored in ``OPENAI_API_KEY``.  An environment
# variable keeps the secret outside the source code.  This line requires an API key.
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# This model changes text into vectors called embeddings.  The same model must be
# used for every product and query so all vectors have matching dimensions.
EMBEDDING_MODEL = "text-embedding-3-small"


# Each dictionary describes one product.  Dictionaries are useful here because
# each value has a clear name, such as ``name`` or ``keywords``.
products = [
	{
		"name": "Wireless Noise-Cancelling Headphones",
		"category": "Audio",
		"description": "Comfortable headphones with clear sound and long battery life.",
		"keywords": ["music", "wireless", "noise cancelling"],
	},
	{
		"name": "Portable Bluetooth Speaker",
		"category": "Audio",
		"description": "A compact speaker for powerful sound at home or outdoors.",
		"keywords": ["music", "portable", "bluetooth"],
	},
	{
		"name": "Smart Fitness Watch",
		"category": "Wearables",
		"description": "Tracks workouts, heart rate, sleep, and daily activity.",
		"keywords": ["fitness", "health", "smartwatch"],
	},
	{
		"name": "Ergonomic Office Chair",
		"category": "Office",
		"description": "An adjustable chair designed for comfortable desk work.",
		"keywords": ["office", "desk", "comfort"],
	},
	{
		"name": "Mechanical Keyboard",
		"category": "Computers",
		"description": "A responsive keyboard for programming and productive work.",
		"keywords": ["typing", "programming", "computer"],
	},
	{
		"name": "USB-C Laptop Dock",
		"category": "Computers",
		"description": "Connects a laptop to monitors, storage, and other accessories.",
		"keywords": ["laptop", "workspace", "usb-c"],
	},
	{
		"name": "Digital Drawing Tablet",
		"category": "Creative Tools",
		"description": "A pressure-sensitive tablet for digital art and illustration.",
		"keywords": ["drawing", "design", "creative"],
	},
]


def create_product_text(product):
	"""Combine one product's fields into one descriptive embedding input.

	``product`` is the function argument: it is the dictionary supplied by the
	caller.  The returned string gives the embedding model all relevant details.
	"""
	# ``join`` combines the keyword list into readable text separated by commas.
	return (
		f"Name: {product['name']}\n"
		f"Category: {product['category']}\n"
		f"Description: {product['description']}\n"
		f"Keywords: {', '.join(product['keywords'])}"
	)


def create_embeddings(texts):
	"""Return one embedding vector for each supplied text.

	``texts`` can be one string or a list of strings.  A list lets us send many
	products in one API request instead of making one request per product.
	"""
	# The API expects a list when processing multiple inputs.  Wrapping one string
	# in a list makes this function work consistently for both input types.
	if isinstance(texts, str):
		texts = [texts]

	# ``embeddings.create`` sends the text to the selected model.  ``model`` tells
	# the API which embedding model to use, and ``input`` contains the text values.
	response = client.embeddings.create(
		model=EMBEDDING_MODEL,
		input=texts,
	)
	# ``response.data`` contains one result per input.  Extract only each vector
	# because the recommendation code does not need the rest of the response.
	return [item.embedding for item in response.data]


def find_n_closest(query_embedding, embeddings, n=3):
	"""Return the indexes and distances of the ``n`` closest embeddings.

	``query_embedding`` is the vector we are searching from.
	``embeddings`` is the collection of product vectors to compare against.
	``n`` controls how many results are returned and defaults to three.
	"""
	# Keep each distance together with its original index so the winning vector
	# can later be matched back to the correct product dictionary.
	distance_scores = [
		{
			"index": index,
			# Cosine distance measures how different two vector directions are.
			# A smaller value means the products are more semantically similar.
			"distance": distance.cosine(query_embedding, embedding),
		}
		# ``enumerate`` supplies both the position and the vector itself.
		for index, embedding in enumerate(embeddings)
	]
	# ``sorted`` orders the results from smallest distance to largest.  ``key``
	# tells it to sort by the value stored under the ``distance`` dictionary key.
	return sorted(distance_scores, key=lambda result: result["distance"])[:n]


# The history represents products the user has already viewed or purchased.
# Indexing ``products`` reuses existing dictionaries instead of copying them.
user_history = [products[0], products[4]]
# ``[-1]`` selects the final item in the list: the most recently viewed product.
last_product = user_history[-1]

# Turn each history dictionary into text, then into a numeric vector.
history_texts = [create_product_text(product) for product in user_history]
history_embeddings = create_embeddings(history_texts)
# ``np.mean`` calculates the average vector.  ``axis=0`` means average each
# vector dimension across all history items, creating one user-interest profile.
mean_history_embedding = np.mean(history_embeddings, axis=0)

# Exclude products already in the user's history from future recommendations.
# ``not in`` compares each complete product dictionary with the history list.
products_filtered = [product for product in products if product not in user_history]
# Create and embed only the products that are still eligible for recommendation.
product_texts = [create_product_text(product) for product in products_filtered]
product_embeddings = create_embeddings(product_texts)

# Recommend unseen products that are closest to the user's average interests.
profile_hits = find_n_closest(mean_history_embedding, product_embeddings)

# Compare the last product with every product, including products already viewed.
last_product_text = create_product_text(last_product)
all_product_texts = [create_product_text(product) for product in products]
# ``[0]`` selects the first (and only) vector returned for one input string.
last_product_embedding = create_embeddings(last_product_text)[0]
all_product_embeddings = create_embeddings(all_product_texts)

# Find the three smallest cosine distances and use their indexes to retrieve products.
hits = find_n_closest(last_product_embedding, all_product_embeddings)

# Print recommendations based on the average of the user's past interests.
print("Recommendations based on your history:")
for hit in profile_hits:
	# ``hit['index']`` points into ``products_filtered``, not the original list.
	recommendation = products_filtered[hit["index"]]
	print(f"- {recommendation['name']} (distance: {hit['distance']:.3f})")

# Print products most similar to the most recently viewed product.
print(f"Because you viewed: {last_product['name']}")
for hit in hits:
	# These indexes point into the complete ``products`` list.
	recommendation = products[hit["index"]]
	print(f"- {recommendation['name']} (distance: {hit['distance']:.3f})")
