# RAG Chatbot

An **Enterprise-style Retrieval-Augmented Generation (RAG) system** that enables users to upload documents or scrape websites and ask questions about the content.
The system retrieves relevant passages using multiple retrieval strategies and generates accurate answers using a Large Language Model.

The project demonstrates a **modular RAG architecture with multi-retrieval, fusion, reranking, and LLM generation**, implemented with Python.

---

# Features

* Upload **PDF documents**
* Scrape content from **websites**
* **Semantic search** using FAISS vector database
* **Multi-retriever architecture**
* **Reciprocal Rank Fusion (RRF)** for combining results
* **Cross-encoder reranking** for improved relevance
* **Groq LLM integration** for fast answer generation
* **Source citations** for retrieved context
* Clean **Streamlit user interface**

---

# System Architecture

The RAG pipeline follows a modular architecture:

1. **Document Ingestion**

   * PDF parsing
   * Web page scraping
   * Text chunking

2. **Embedding Generation**

   * Sentence transformer embeddings
   * Vector representation of document chunks

3. **Vector Storage**

   * FAISS vector index
   * Efficient similarity search

4. **Multi-Retrieval**

   * Vector retriever
   * Sentence window retriever
   * Graph-based retriever

5. **Fusion**

   * Reciprocal Rank Fusion (RRF) merges retriever results

6. **Reranking**

   * Cross-encoder model improves ranking accuracy

7. **Answer Generation**

   * Groq-hosted LLM generates answers using retrieved context

---

# Project Structure

```
enterprise-rag-system/
│
├── app.py
├── config.py
├── requirements.txt
├── start.sh
├── .env.example
├── README.md
│
├── embeddings/
├── vectorstore/
├── ingestion/
├── retrievers/
├── fusion/
├── reranker/
├── llm/
├── langgraph_pipeline/
└── utils/
```

---

# Installation

Clone the repository:

```
git clone https://github.com/arifekbalrashid/RAG_chatbot.git
cd RAG_chatbot
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file based on `.env.example`.

Example:

```
GROQ_API_KEY=your_api_key_here
EMBEDDING_MODEL=BAAI/bge-small-en
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
GROQ_MODEL=llama-3.3-70b-versatile
```

---

# Running the Application

Start the Streamlit app:

```
streamlit run app.py
```

Then open the app in your browser.

---

# Usage

1. Upload a **PDF document** or provide a **website URL**
2. The system processes and indexes the content
3. Ask questions related to the uploaded information
4. The system retrieves relevant passages and generates an answer

---

# Technologies Used

* Python
* Streamlit
* FAISS
* LangGraph
* Sentence Transformers
* Cross Encoder Reranking
* Groq LLM API
* BeautifulSoup
* Requests

---

# Future Improvements

Possible extensions for the system:

* Hybrid search (BM25 + vector)
* Persistent vector storage
* Chat history with memory
* Streaming LLM responses
* Multi-document collections
* Authentication and user management

