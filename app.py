import os
os.environ["STREAMLIT_HOME"] = os.getcwd()

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# Split text into chunks
def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Build FAISS index
def build_faiss_index(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Retrieve most relevant chunk
def retrieve_answer(query, chunks, index):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=1)
    return chunks[indices[0][0]]

# ---------------- Streamlit App ---------------- #
st.set_page_config(page_title="PDF Chatbot", layout="wide")

st.title("PDF Chatbot")
st.write("Upload a PDF document and ask questions about its content.")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)
        chunks = split_text(text)
        index, _ = build_faiss_index(chunks)
        st.success("PDF processed successfully.")

    # Chat section
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question:")

    if query:
        answer = retrieve_answer(query, chunks, index)
        st.markdown(f"**Answer:** {answer}")
