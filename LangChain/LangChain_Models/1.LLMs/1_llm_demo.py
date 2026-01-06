from langchain_openai import OpenAI
from dotenv import load_dotenv 
# this is use to read the .env file open ai key 


load_dotenv()

llm = OpenAI(model='openai/gpt-5.2')

# this is help to ask the question to the models 
result =llm.invoke("What is the capital of india ")

print(result)