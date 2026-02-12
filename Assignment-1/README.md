# RAG (Retrieval-Augmented Generation) Assignment

A Question-Answering system built using RAG architecture that answers questions based on PDF documents.

## 📋 Project Overview

This project implements a complete RAG pipeline that:
1. **Loads** PDF documents from a data folder
2. **Chunks** documents using semantic-aware splitting
3. **Embeds** text chunks using Google's embedding model
4. **Stores** vectors in FAISS for efficient retrieval
5. **Retrieves** relevant context for user queries
6. **Generates** accurate answers using Google Gemini LLM

## 🏗️ RAG Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAG PIPELINE ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

                              INDEXING PHASE
┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐
│  PDF     │───▶│   Text       │───▶│   Text      │───▶│   Embedding    │
│  Docs    │    │   Extraction │    │   Chunking  │    │   Generation   │
└──────────┘    └──────────────┘    └─────────────┘    └────────────────┘
                                                               │
                                                               ▼
                                                       ┌────────────────┐
                                                       │  FAISS Vector  │
                                                       │    Store       │
                                                       └────────────────┘
                                                               │
                              RETRIEVAL PHASE                  │
┌──────────┐    ┌──────────────┐    ┌─────────────┐           │
│  User    │───▶│   Query      │───▶│  Similarity │◀──────────┘
│  Query   │    │   Embedding  │    │   Search    │
└──────────┘    └──────────────┘    └─────────────┘
                                           │
                                           ▼
                              GENERATION PHASE
                       ┌─────────────────────────┐
                       │   Top-K Relevant Chunks │
                       └─────────────────────────┘
                                   │
                                   ▼
                       ┌─────────────────────────┐
                       │   Google Gemini LLM     │
                       └─────────────────────────┘
                                   │
                                   ▼
                       ┌─────────────────────────┐
                       │   Generated Response    │
                       └─────────────────────────┘
```

## 🛠️ Tools & Libraries

| Library | Purpose |
|---------|---------|
| LangChain | RAG orchestration framework |
| langchain-google-genai | Google Gemini integration |
| FAISS | Vector similarity search |
| PyPDF | PDF text extraction |
| Streamlit | Web UI (bonus) |
| python-dotenv | Environment management |

## 📊 Configuration Details

### Text Chunking Strategy
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters
- **Splitter**: RecursiveCharacterTextSplitter
- **Reason**: Preserves semantic coherence while maintaining optimal chunk size for retrieval

### Embedding Model
- **Model**: Google `models/embedding-001`
- **Dimension**: 768
- **Reason**: High-quality embeddings with free tier, seamless Gemini integration

### Vector Database
- **Database**: FAISS (Facebook AI Similarity Search)
- **Reason**: Fast, memory-efficient, no external dependencies

### LLM
- **Model**: Gemini 1.5 Flash
- **Temperature**: 0.3 (for factual responses)
- **Reason**: Cost-effective, fast, high-quality responses

## 📁 Project Structure

```
Assignment-1/
├── data/                    # PDF documents folder
├── vectorstore/             # FAISS index storage
├── rag_assignment.ipynb     # Main notebook with step-by-step implementation
├── app.py                   # Streamlit web interface
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
├── .env                     # Your API key (create this)
└── README.md                # This file
```

## 🚀 Instructions to Run

### Prerequisites
- Python 3.9+
- Google API Key (get from [Google AI Studio](https://makersuite.google.com/app/apikey))

### Setup

1. **Clone/Navigate to the project**
   ```bash
   cd Assignment-1
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   ```bash
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

5. **Add PDF documents**
   - Place your PDF files in the `data/` folder

### Run Jupyter Notebook

```bash
jupyter notebook rag_assignment.ipynb
```

### Run Streamlit UI (Bonus)

```bash
streamlit run app.py
```

## 🧪 Test Queries

The notebook includes these test queries:

1. **"What is the main topic of the document?"**
   - Tests basic comprehension and summarization

2. **"Can you summarize the key points discussed?"**
   - Tests multi-chunk retrieval and synthesis

3. **"What are the conclusions or recommendations mentioned?"**
   - Tests specific information extraction

## 🔮 Future Improvements

### 1. Better Chunking
- Semantic chunking using sentence embeddings
- Document structure-aware splitting
- Variable overlap based on content density

### 2. Reranking / Hybrid Search
- Cross-encoder reranking for better relevance
- Combine dense (embedding) + sparse (BM25) retrieval
- Maximal Marginal Relevance (MMR) for diversity

### 3. Metadata Filtering
- Filter by document, date, or section
- Support for document categorization
- Custom metadata extraction

### 4. UI Enhancements
- Chat history with conversation context
- Document upload feature
- Source highlighting in PDF viewer

### 5. Advanced Features
- Query expansion and rephrasing
- Multi-modal RAG (images, tables)
- Response caching for efficiency
- RAGAS evaluation metrics

## 📝 Summary Table

| Component | Choice | Reason |
|-----------|--------|--------|
| Document Loader | PyPDFLoader | Reliable PDF text extraction |
| Text Splitter | RecursiveCharacterTextSplitter | Semantic coherence preservation |
| Chunk Size | 1000 chars | Balance between context and precision |
| Chunk Overlap | 200 chars | Continuity between chunks |
| Embedding Model | Google embedding-001 | High quality, free tier available |
| Vector Store | FAISS | Fast, local, memory efficient |
| LLM | Gemini 1.5 Flash | Cost-effective, fast responses |
| Chain Type | Stuff | Simple, effective for small context |

## 👤 Author

Assignment submitted for Agentic AI Course

---

**Note**: Make sure to add your own PDF documents to the `data/` folder before running the notebook or Streamlit app.
