from langchain_openai import ChatOpenAI
# here we import chatOpneAi model 

from dotenv import load_dotenv

load_dotenv()



model =ChatOpenAI(model='gpt-4',temperature=1.8, max_completion_tokens=10)
# here the temperature decides the randomness of the models 
# max_completion_tokens to restrict the tokens 

result = model.invoke("What is the capital of india ? ")

print(result)
print(result.content)