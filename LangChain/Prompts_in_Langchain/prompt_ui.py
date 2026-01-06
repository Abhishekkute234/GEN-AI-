from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

# Initialize HuggingFace Endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
    top_k=50
)

# Wrap with ChatHuggingFace
model = ChatHuggingFace(llm=llm)

# Streamlit UI
st.header("🔬 Research Paper Explanation Tool")

# Input selections
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Select...",
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis",
        "Types of language in the coding world"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

user_input = st.text_area(
    "Enter your research query here:",
    placeholder="e.g., Explain the transformer architecture in simple terms"
)

# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["paper", "style", "length", "query"],
    template="""
You are an expert research assistant explaining academic papers.

**Research Paper:** {paper}
**Explanation Style:** {style}
**Length:** {length}
**User Query:** {query}

Please provide a clear and accurate explanation based on the selected style and length.
If no specific query is provided, give a comprehensive overview of the paper.

Response:
"""
)

# Button to generate explanation
if st.button('Generate Explanation'):
    if paper_input == "Select...":
        st.warning("⚠️ Please select a research paper.")
    elif not user_input.strip():
        st.warning("⚠️ Please enter a research query.")
    else:
        with st.spinner('Generating explanation...'):
            try:
                # Format the prompt with user inputs
                formatted_prompt = prompt_template.format(
                    paper=paper_input,
                    style=style_input,
                    length=length_input,
                    query=user_input
                )
                
                # Invoke the model
                result = model.invoke(formatted_prompt)
                
                # Display the result
                st.markdown("### 📝 Explanation:")
                st.write(result.content)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure your HUGGINGFACEHUB_API_TOKEN is set in your .env file")

# Sidebar with information
with st.sidebar:
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool uses **HuggingFace's Zephyr-7B-Beta** model to explain 
    research papers in different styles and lengths.
    
    **How to use:**
    1. Select a research paper
    2. Choose your preferred explanation style
    3. Select the length of explanation
    4. Enter your specific query
    5. Click 'Generate Explanation'
    
    **Note:** Requires HUGGINGFACEHUB_API_TOKEN in .env file
    """)