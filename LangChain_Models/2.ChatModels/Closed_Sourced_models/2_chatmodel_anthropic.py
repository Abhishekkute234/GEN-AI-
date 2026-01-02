 # Install: pip install langchain-anthropic

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key="YOUR_API_KEY")
result = llm.invoke("What is the capital of India?")
print(result)