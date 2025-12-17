# LLM Document Intelligence

An AI-powered application that allows users to upload documents and interact with them using Large Language Models.

## Features
- Upload PDF or text documents
- Automatic document summarization
- Question & Answering over documents
- Vector search using FAISS

## Tech Stack
- Python
- LangChain
- OpenAI
- FAISS
- Streamlit

## How It Works
1. Document is split into chunks
2. Embeddings are generated and stored in FAISS
3. LLM retrieves relevant chunks to answer user queries

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
