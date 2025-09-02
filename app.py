import streamlit as st
import os
import tempfile
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------
# Embedding & QA Models
# -------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# -------------------------
# Utility Functions
# -------------------------
def load_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def split_text(text, chunk_size=500):
    """Split text into smaller chunks for embedding."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(chunks):
    """Build FAISS index for chunk embeddings."""
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def generate_answer(query, chunks, faiss_index, mode="concise"):
    """Generate final answer using retriever + QA model."""
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    scores, indices = faiss_index.search(query_vec, 3)
    relevant_chunks = [chunks[i] for i in indices[0]]
    context = " ".join(relevant_chunks)

    try:
        result = qa_pipeline(question=query, context=context)
        answer = result["answer"]
    except Exception:
        answer = "Sorry, I couldn’t find an exact answer."

    if mode == "concise":
        return answer
    else:
        return f"Answer: {answer}\n\nContext: {context}"

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("📄 PDF Q&A Bot")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        text = load_pdf(tmp_path)
        chunks = split_text(text)
        faiss_index, _ = build_faiss_index(chunks)

        st.success("✅ PDF processed successfully!")

        query = st.text_input("Ask a question about your PDF:")
        mode = st.radio("Answer Mode:", ["concise", "deep"])

        if query:
            answer = generate_answer(query, chunks, faiss_index, mode.lower())
            st.markdown("### Answer (from PDF):")
            st.write(answer)

if __name__ == "__main__":
    main()
