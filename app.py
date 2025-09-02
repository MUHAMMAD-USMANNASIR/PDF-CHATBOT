import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ---------------- Setup ---------------- #
st.set_page_config(page_title="PDF Chatbot", layout="wide")

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return embedder, summarizer

model, summarizer = load_models()

# ---------------- PDF Handling ---------------- #
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def split_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ---------------- Embedding + Retrieval ---------------- #
def build_faiss_index(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_answer(query, chunks, index, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    combined = " ".join(results)

    # Summarize retrieved content into short answer
    summary = summarizer(combined, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
    return summary

# ---------------- Streamlit App ---------------- #
st.title("PDF Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"], key="pdf_upload")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)

        if not text:
            st.error("No extractable text found in this PDF. Try another file.")
        else:
            chunks = split_text(text)
            index = build_faiss_index(chunks)
            st.success("PDF processed successfully.")

            st.subheader("Ask a Question")
            query = st.text_input("Enter your question:", key="user_query")

            if query:
                answer = retrieve_answer(query, chunks, index, top_k=3)
                st.markdown("**Answer:**")
                st.write(answer)
