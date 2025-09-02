import os
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- Setup ---------------- #
st.set_page_config(page_title="PDF Chatbot", layout="wide")

@st.cache_resource
def load_model():
    """Load sentence transformer model (cached)."""
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- PDF Handling ---------------- #
def extract_text_from_pdf(pdf_file):
    """Extract raw text from PDF."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()

def split_text(text, chunk_size=200):
    """Split text into smaller chunks of words."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ---------------- Embedding + Retrieval ---------------- #
def build_faiss_index(chunks):
    """Build FAISS index for chunks."""
    if not chunks:
        return None, None
    embeddings = model.encode(chunks, convert_to_numpy=True)
    if embeddings.shape[0] == 0:
        return None, None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_answer(query, chunks, index, top_k=2):
    """Retrieve most relevant chunk(s) from PDF."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k=top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return " ".join(results)

# ---------------- Streamlit App ---------------- #
st.title("PDF Chatbot")
st.write("Upload a PDF document and ask questions about its content.")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"], key="pdf_upload")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(uploaded_file)

        if not text:
            st.error("No extractable text found in this PDF. Try another file.")
        else:
            chunks = split_text(text)
            index, _ = build_faiss_index(chunks)

            if index is None:
                st.error("Failed to process this PDF.")
            else:
                st.success("PDF processed successfully.")

                st.subheader("Ask a Question")
                query = st.text_input("Enter your question:", key="user_query")

                if query:
                    answer = retrieve_answer(query, chunks, index, top_k=2)
                    st.markdown("**Answer:**")
                    st.write(answer)
