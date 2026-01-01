# LangChains

# LangChain & GenAI Study Notes

A comprehensive guide covering LangChain components, Generative AI concepts, and practical implementation patterns for building LLM-powered applications.

---

## Table of Contents

1. [LangChain Components](#langchain-components)
2. [Generative AI Overview](#generative-ai-overview)
3. [LangChain Framework](#langchain-framework)
4. [Memory Management](#memory-management)
5. [Chains](#chains)
6. [Indexers](#indexers)

---

## LangChain Components

LangChain provides six core components for building AI applications:

### 1. **Models**

Core interfaces for interacting with AI models:

- **Language Models (LLMs)**: Text generation (e.g., GPT, Claude)
- **Embedding Models**: Convert text to vector representations for semantic search

### 2. **Prompts**

Inputs provided to LLMs that guide the model's response:

- **Examples**: "Summarize this topic", "Use an informative tone"
- **Types**:
  - **Dynamic & Reusable Prompts**: Parameterized templates
  - **Role-Based Prompts**: Like asking doctors about fever symptoms
  - **Few-Shot Prompting**: Provide examples with customer support templates

### 3. **Chains**

Pipelines that connect LLMs and process data sequentially:

- Output of one stage becomes input of the next
- Enable complex workflows
- **Examples**:
  - Input → LLM → Translator → Summary
  - Input → Prepare → Report → Combine → Output (Parallel chains)
  - AI agent feedback loops (Good → Thank you, Bad → Email/Form)

### 4. **Memory**

LLM API calls are stateless - memory manages conversation context:

**Types of Memory**:

- **Conversation Buffer Memory**: Adds chat history and processes the entire history
- **Conversation Window Memory**: Only keeps the last N messages
- **Summarizer**: Summarizes the chat history
- **Custom Memory**: Advanced use cases

**Key Points**:

- No memory of last question by default
- "He" in follow-up questions won't decode without memory
- Helps maintain context across interactions

### 5. **Agents**

Build AI agents that can:

- Use LLMs with text generation capabilities
- Act as chatbots with memory
- Examples: Travel website booking (flights, hotels, work arrangements)

### 6. **Indexers**

Connect applications to external knowledge sources:

- PDFs, websites, databases
- **Components**:
  - **Document Loaders**: Load external documents
  - **Text Splitters**: Break documents into chunks
  - **Vector Stores**: Store embeddings for semantic search
  - **Retrievers**: Query and retrieve relevant information

---

## Generative AI Overview

**Definition**: AI that creates new content (text, images, music, code) by learning patterns from existing data and mimicking human creativity.

### Impact Areas

- Customer Support
- Content Creation
- Education
- Software Development

### Mental Model

#### Foundation Model

Two perspectives on building AI applications:

**User Perspective**:

- Prompt Engineering
- RAG (Retrieval-Augmented Generation)
- AI Agents
- Vector Databases
- Workflow/Builder perspective

**Builder Perspective**:

- RLHF (Reinforcement Learning from Human Feedback)
- Pretraining
- Quantization
- Fine-tuning:
  - Task-specific tuning
  - Instruction tuning
  - Continual pretraining
- Evaluation → Deployment

### Transformer Architecture

**Types**:

- **Encoder Only (BERT)**: Understanding tasks
- **Decoder Only (GPT)**: Generation tasks
- **Encoder-Decoder (T5)**: Translation and complex tasks

### Training Process

- Training data tokenization
- Training strategy
- Handling challenges
- Evaluation
- Optimization: Training optimization, model compression, inference optimization

---

## LangChain Framework

**Definition**: Open-source framework for developing applications powered by LLMs.

### Key Features

- End-to-end tool for building LLM-based applications
- Supports all major LLMs
- LLM-based application development
- Free to use
- Integrates with major tools
- Supports all major GenAI use cases

### Use Cases

- Users' perspective: Building basic LLM apps
- Open-source vs closed-source LLMs
- Using LangChain with face APIs
- RAG implementation
- Prompt engineering
- Fine-tuning
- Agentic workflows

### Real-World Applications

- Conversational chatbots
- AI knowledge assistants
- AI agents
- Workflow automation
- Summarization/research helpers

---

## Memory Management

### Semantic Search Process

Understanding the meaning of queries through:

1. Convert query to vector representation
2. Use paragraph about topic (e.g., Nike)
3. Store runs/visits information
4. Compare with Rohit Sharma example
5. Convert to vector using chunking (JB, VK, RS)

### How PDF Processing Works

Example workflow:

1. Upload PDF to AWS S3 component
2. Document loader retrieves content
3. Split into pages (Page 1, Page 2, etc.)
4. Process with MaybeChain or vector format
5. Pages undergo:
   - Embedding 1
   - Embedding 2
   - Embedding (for page n)
6. Store in vector database with company vectors
7. User query → Embedding → Compare embeddings → Semantic search
8. Return system with: Page 5 + User query → Brain (LLM API) → Final output

**Note**: This whole interface generation is done by LangChain itself. You just need to implement your ideas.

### What LangChain Builds

- Conversational chatbots
- AI knowledge assistants
- AI agents
- Workflow automation
- Summarization/research helpers

---

## Chains

Pipelines connecting LLMs for sequential processing.

### Basic Pipeline Example

```
Input (English text, 1000 words)
  → [LLM]
  → Translator
  → [Summary (100 words)]
```

_Note: Manual implementation required_

### Complex Pipeline Features

- Make output of first stage the input of next stage
- Build complex pipelines
- Examples with divisions/reports

### Parallel Chain Example

```
[Input] → [Prepare] → [Report] → [Combine] → [Output]
         → [Summarize]
```

### AI Agent Feedback (Conditional Chain)

```
[Input/Feedback] → [Process]
                      ↓
                   [Good] → [Thank you]
                      ↓
                   [Bad] → [Email form]
```

---

## Indexers

Indexers connect your application to external knowledge sources like PDFs, websites, or databases.

### Components

#### 1. **Document Loaders**

- LLM-1: External + knowledge (e.g., PAGE, PROP)
- Load documents from various sources

#### 2. **Text Splitters**

Break documents into manageable chunks

#### 3. **Vector Stores**

Store embeddings for efficient retrieval

#### 4. **Retrievers**

- ChatGPT → XYZ → Rule
- Normalize of company
- Personal question feature
- Leave policy of my XYZ company

Query and retrieve relevant information from the knowledge base.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Basic understanding of LLMs
- API keys for chosen LLM providers

### Installation

```bash
pip install langchain
pip install langchain-openai  # or other LLM providers
pip install chromadb  # for vector storage
pip install tiktoken  # for tokenization
```

### Basic Usage Example

```python
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize your LLM
llm = OpenAI(temperature=0.7)

# Create a prompt template
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text: {text}"
)

# Build a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Execute
result = chain.run(text="Your long text here...")
print(result)
```

### RAG Implementation Example

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Query
response = qa_chain.run("What is the document about?")
```

### Memory Implementation Example

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory
memory = ConversationBufferMemory()

# Create conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat
response1 = conversation.predict(input="Hi, my name is John")
response2 = conversation.predict(input="What's my name?")
# Output: "Your name is John"
```

---

## Architecture Patterns

### Pattern 1: Simple LLM Chain

```
User Input → Prompt Template → LLM → Output
```

### Pattern 2: RAG Pattern

```
User Query → Embedding → Vector Search → Relevant Docs + Query → LLM → Answer
```

### Pattern 3: Agent Pattern

```
User Request → Agent → Tool Selection → Tool Execution → Agent → Response
```

### Pattern 4: Sequential Chain

```
Input → Chain 1 → Chain 2 → Chain 3 → Final Output
```

---

## Best Practices

1. **Prompt Engineering**:

   - Start with clear, specific prompts
   - Use examples for few-shot learning
   - Define desired output format

2. **Memory Management**:

   - Choose appropriate memory type for your use case
   - Consider token limits
   - Use summarization for long conversations

3. **Chain Design**:

   - Keep chains modular and testable
   - Use parallel chains when appropriate
   - Handle errors gracefully

4. **Vector Storage**:

   - Optimize chunk sizes (1000-2000 tokens)
   - Use overlap for context preservation
   - Choose appropriate embedding model

5. **Agent Design**:
   - Define clear tools and goals for agents
   - Limit tool complexity
   - Monitor agent behavior

---

## Common Use Cases

### 1. Chatbot with Memory

Build conversational AI that remembers context

### 2. Document Q&A System

Query PDFs, websites, or documents using RAG

### 3. Content Generation

Create blog posts, summaries, translations

### 4. Code Assistant

Help with coding tasks and explanations

### 5. Data Analysis

Analyze and visualize data with natural language

---

## Troubleshooting

### Common Issues

**Issue**: Token limit exceeded

- **Solution**: Use text splitters and chunking

**Issue**: Poor retrieval results

- **Solution**: Adjust chunk size and overlap, improve embeddings

**Issue**: Agent loops infinitely

- **Solution**: Add iteration limits and better tool descriptions

**Issue**: Memory grows too large

- **Solution**: Use conversation window or summary memory

---

## Resources

- [LangChain Documentation](https://docs.langchain.com)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [Anthropic Prompt Engineering](https://docs.anthropic.com)
- [OpenAI API Documentation](https://platform.openai.com/docs)

---

## Project Structure

```
project/
│
├── notebooks/
│   ├── 01_basic_chains.ipynb
│   ├── 02_memory_examples.ipynb
│   ├── 03_rag_implementation.ipynb
│   └── 04_agents.ipynb
│
├── src/
│   ├── chains/
│   ├── agents/
│   ├── utils/
│   └── config.py
│
├── data/
│   ├── documents/
│   └── vectorstore/
│
├── requirements.txt
└── README.md
```

---



