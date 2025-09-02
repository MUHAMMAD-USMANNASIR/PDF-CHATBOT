import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Load local QA model
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

qa_pipeline = load_qa_model()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Split text into chunks
def split_into_chunks(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Build FAISS index
def build_faiss_index(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, chunks

# Search top passages
def search(query, index, embeddings, chunks, top_k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)

    results = []
    for i, dist in zip(I[0], D[0]):
        if i == -1:
            continue
        results.append(chunks[i])
    return results if results else ["No relevant passages found."]

# Generate answer using QA model
def generate_answer(question, passages):
    context = " ".join(passages)  # merge top passages
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

# ---------------- UI ----------------
st.title("📄 PDF Chatbot")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(pdf_file)
        chunks = split_into_chunks(text)
        index, embeddings, chunks = build_faiss_index(chunks)

    st.success("PDF indexed successfully. You can now ask questions.")

    query = st.text_input("Enter your question:")

    if query:
        passages = search(query, index, embeddings, chunks, top_k=3)
        answer = generate_answer(query, passages)

        st.subheader("Answer:")
        st.write(answer)
