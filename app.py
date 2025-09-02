import streamlit as st
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ---------------------
# Load models
# ---------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# ---------------------
# Functions
# ---------------------

def extract_text_from_pdf(file):
    """Extract raw text from uploaded PDF"""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=400, overlap=100):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_faiss_index(chunks):
    """Create FAISS index of embeddings"""
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def answer_question(query, chunks, index, embeddings, top_k=3):
    """Retrieve context and run QA pipeline"""
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(query_emb, dtype="float32"), top_k)

    # Always take top passages
    retrieved_passages = [chunks[i] for i in I[0] if i < len(chunks)]
    context = " ".join(retrieved_passages)

    if not context.strip():
        return "Sorry, I couldn’t find anything relevant.", 0.0

    result = qa_pipeline(question=query, context=context)
    return result["answer"], float(result["score"])

# ---------------------
# Streamlit App
# ---------------------
st.title("PDF Chatbot")
st.write("Ask questions about your uploaded PDF.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)

    st.success("PDF processed. You can now ask questions.")

    query = st.text_input("Enter your question:")
    if query:
        answer, confidence = answer_question(query, chunks, index, embeddings)
        st.subheader("Answer:")
        st.write(answer)
        st.caption(f"Confidence: {confidence:.2f}")
