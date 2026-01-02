from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Fast and free
    google_api_key="YOUR_API_KEY_HERE",
    temperature=0.7
)

# Use it
result = llm.invoke("What is the capital of India?")
print(result.content)