"""
RAG Application - Streamlit UI
A web interface for the RAG-based Question Answering System.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Configuration
DATA_PATH = "data/"
VECTOR_STORE_PATH = "vectorstore/faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# Page configuration
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F77B4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 1.1rem;
    }
    .answer-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system components."""
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ GOOGLE_API_KEY not found. Please set it in .env file")
        return None, None
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Check if vector store exists
    if os.path.exists(VECTOR_STORE_PATH):
        vector_store = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return embeddings, vector_store
    
    return embeddings, None


def load_and_process_documents():
    """Load PDFs and create vector store."""
    with st.spinner("Loading documents..."):
        loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        documents = loader.load()
    
    if not documents:
        st.warning("⚠️ No PDF documents found in the data/ folder")
        return None
    
    with st.spinner("Chunking documents..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
    
    with st.spinner("Creating embeddings..."):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        vector_store.save_local(VECTOR_STORE_PATH)
    
    return vector_store


def create_qa_chain(vector_store):
    """Create the QA chain."""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        max_tokens=1024
    )
    
    prompt_template = """
    You are a helpful assistant that answers questions based on the provided context.
    Use ONLY the information from the context to answer the question.
    If the answer is not found in the context, say "I cannot find this information in the provided documents."

    Context:
    {context}

    Question: {question}

    Answer: Provide a clear, concise answer based on the context above.
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain


def main():
    """Main application."""
    st.markdown('<h1 class="main-header">📚 RAG Question & Answer System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")
        
        st.subheader("📊 RAG Settings")
        st.write(f"**Chunk Size:** {CHUNK_SIZE}")
        st.write(f"**Chunk Overlap:** {CHUNK_OVERLAP}")
        st.write(f"**Top-K Results:** {TOP_K}")
        
        st.markdown("---")
        
        st.subheader("📁 Document Management")
        if st.button("🔄 Reindex Documents", use_container_width=True):
            with st.spinner("Reindexing..."):
                vector_store = load_and_process_documents()
                if vector_store:
                    st.success("✅ Documents reindexed successfully!")
                    st.rerun()
        
        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown("""
        This RAG system uses:
        - **LangChain** for orchestration
        - **Google Gemini** for LLM
        - **FAISS** for vector storage
        - **PyPDF** for PDF parsing
        """)
    
    # Initialize system
    embeddings, vector_store = initialize_rag_system()
    
    if vector_store is None:
        st.warning("📂 No indexed documents found. Please add PDF files to the `data/` folder and click 'Reindex Documents'.")
        
        # Check for PDFs in data folder
        if os.path.exists(DATA_PATH):
            pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
            if pdf_files:
                st.info(f"Found {len(pdf_files)} PDF file(s). Click 'Reindex Documents' to process them.")
        return
    
    # Create QA chain
    qa_chain = create_qa_chain(vector_store)
    
    # Query input
    st.subheader("🔍 Ask a Question")
    query = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know about the documents?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🚀 Search", use_container_width=True)
    with col2:
        show_sources = st.checkbox("Show source documents", value=True)
    
    # Process query
    if search_button and query:
        with st.spinner("Searching and generating answer..."):
            try:
                response = qa_chain.invoke({"query": query})
                
                # Display answer
                st.subheader("💡 Answer")
                st.markdown(f'<div class="answer-box">{response["result"]}</div>', unsafe_allow_html=True)
                
                # Display sources
                if show_sources and "source_documents" in response:
                    st.subheader(f"📚 Sources ({len(response['source_documents'])} documents)")
                    for i, doc in enumerate(response["source_documents"], 1):
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "N/A")
                        with st.expander(f"Source {i}: {os.path.basename(source)} (Page {page})"):
                            st.write(doc.page_content)
                            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Example queries
    st.markdown("---")
    st.subheader("💡 Example Queries")
    example_queries = [
        "What is the main topic of the document?",
        "Can you summarize the key points?",
        "What are the conclusions mentioned?"
    ]
    
    cols = st.columns(len(example_queries))
    for col, example in zip(cols, example_queries):
        with col:
            if st.button(example, use_container_width=True):
                st.session_state["query"] = example
                st.rerun()


if __name__ == "__main__":
    main()
