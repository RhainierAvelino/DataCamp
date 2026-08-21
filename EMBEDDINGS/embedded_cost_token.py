import tiktoken

enc = tiktoken.get_encoding("text-embedding-3-small")

total_tokens = sum(len(enc.encode(text)) for text in documents)  # Count the total number of tokens in the documents

cost_per_1k_tokens = 0.0004  # Cost per 1,000 tokens for the text-embedding-3-small model

print(f"Total tokens: {total_tokens}")
total_cost = (total_tokens / 1000) * cost_per_1k_tokens