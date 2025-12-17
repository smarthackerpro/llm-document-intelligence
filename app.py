
import streamlit as st
from dotenv import load_dotenv
from utils import load_pdf, create_vector_store, get_qa_chain

load_dotenv()

st.set_page_config(page_title="LLM Document Intelligence", layout="wide")

st.title("📄 LLM-Powered Document Intelligence")
st.write("Upload a document, get a summary, and ask questions using AI.")

uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded_file:
    if uploaded_file.type == "application/pdf":
        text = load_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    st.success("Document loaded successfully!")

    with st.spinner("Creating embeddings..."):
        vectorstore = create_vector_store(text)
        qa_chain = get_qa_chain(vectorstore)

    st.subheader("📌 Document Summary")
    summary_prompt = "Summarize this document in 5 bullet points."
    summary = qa_chain.run(summary_prompt)
    st.write(summary)

    st.subheader("💬 Ask Questions")
    question = st.text_input("Ask a question about the document")

    if question:
        answer = qa_chain.run(question)
        st.write("**Answer:**")
        st.write(answer)
