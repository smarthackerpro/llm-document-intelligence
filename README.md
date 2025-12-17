# LLM Document Intelligence 📄🤖

An LLM-powered document intelligence application that allows users to upload documents and ask questions or generate summaries using a Retrieval-Augmented Generation (RAG) pipeline.

---

## 🚀 Features
- Upload PDF or TXT documents
- Automatic text extraction and chunking
- Semantic search using FAISS vector database
- Question Answering and Summarization
- Fully local LLM inference (no paid APIs)

---

## 🧠 Architecture
1. Document is uploaded via Streamlit UI
2. Text is extracted and split into chunks
3. Embeddings are generated using sentence-transformers
4. FAISS stores embeddings for similarity search
5. Relevant chunks are retrieved
6. A local LLM generates grounded answers

---

## 🛠️ Tech Stack
- Python
- Streamlit
- LangChain
- FAISS
- Hugging Face Transformers
- sentence-transformers
- PyTorch

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/smarthackerpro/llm-document-intelligence.git
cd llm-document-intelligence

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python -m streamlit run app.py
