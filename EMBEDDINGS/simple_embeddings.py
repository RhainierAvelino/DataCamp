from openai import OpenAI
import os


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

response = client.embeddings.create(
    model="text-embedding-3-large",
    input="The quick brown fox jumped over the lazy dog."
)

response_dict = response.model_dump()  # Convert the response to a dictionary
print(response_dict)