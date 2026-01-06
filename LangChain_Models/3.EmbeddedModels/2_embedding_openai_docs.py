from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

document = [
  "Delhi is the capital of India.",
  "KOLKATA is the capital of West Bengal.",
  "Mumbai is the capital of Maharashtra.",
  "patna is the capital of Bihar."
]

result = embedding.embed_documents(document)

print(str(result))