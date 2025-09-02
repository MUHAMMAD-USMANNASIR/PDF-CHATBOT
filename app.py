import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer, util
import torch

# Load embedding model once (small, memory-friendly)
@st.cache_resource
def load_embeddings():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Split text into chunks
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Search top-k relevant chunks
def search_chunks(query, chunks, embeddings, top_k=3):
    chunk_embs = embeddings.encode(chunks, convert_to_tensor=True)
    query_emb = embeddings.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_emb, chunk_embs, top_k=top_k)[0]
    return [chunks[hit['corpus_id']] for hit in hits]

# Generate answers
def generate_answer(query, chunks, embeddings, mode="concise"):
    relevant_chunks = search_chunks(query, chunks, embeddings, top_k=3)
    context = " ".join(relevant_chunks)

    if mode == "concise":
        return f"Answer (concise): Based on the PDF, {context[:300]}..."
    elif mode == "deep":
        return f"Answer (deep): The document discusses in detail: {context}"
    else:
        return "Invalid mode selected."

# -------------------- Streamlit UI -------------------- #
def main():
    st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
    st.title("📘 PDF Q&A Chatbot")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        with st.spinner("Reading and processing PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)
            embeddings = load_embeddings()

        st.success("✅ PDF processed! Ask questions below:")

        query = st.text_input("Enter your question:")
        mode = st.radio("Select answer mode:", ["concise", "deep"], horizontal=True)

        if query:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, chunks, embeddings, mode.lower())
            st.subheader("💡 Answer")
            st.write(answer)

if __name__ == "__main__":
    main()
