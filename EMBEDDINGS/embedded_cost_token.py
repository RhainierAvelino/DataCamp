"""Estimate the token count and embedding cost before sending a request."""

import tiktoken


EMBEDDING_MODEL = "text-embedding-3-small"
# Check the OpenAI pricing page when using this as a real cost estimate.
COST_PER_MILLION_TOKENS = 0.02

documents = [
    "Embeddings represent the meaning of text as numbers.",
    "Cosine similarity can compare the meaning of two embeddings.",
    "A vector database stores embeddings for efficient retrieval.",
]

# This tokenizer matches the model family and converts text into tokens.
encoder = tiktoken.encoding_for_model(EMBEDDING_MODEL)
total_tokens = sum(len(encoder.encode(document)) for document in documents)
estimated_cost = total_tokens / 1_000_000 * COST_PER_MILLION_TOKENS

print(f"Total tokens: {total_tokens}")
print(f"Estimated embedding cost: ${estimated_cost:.8f}")
