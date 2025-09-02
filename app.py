import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# -------------------------------
# Load models
# -------------------------------
@st.cache_resource
def load_models():
    qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    deep_model = pipeline("text2text-generation", model="google/flan-t5-small")
    emb_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    return qa_model, deep_model, emb_model

qa_model, deep_model, emb_model = load_models()

# -------------------------------
# PDF text extraction
# -------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text")
    return text

def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# -------------------------------
# Embedding functions
# -------------------------------
def embed_chunks(chunks):
    embeddings = []
    for ch in chunks:
        emb = emb_model(ch)[0][0]  # extract vector
        embeddings.append(np.mean(emb, axis=0))  # pool to single vector
    return np.array(embeddings)

def search_chunks(query, chunks, embeddings, top_k=5):
    query_emb = np.mean(emb_model(query)[0][0], axis=0)
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx]

# -------------------------------
# Answer generation
# -------------------------------
def generate_answer(query, chunks, embeddings, mode="concise"):
    relevant_chunks = search_chunks(query, chunks, embeddings, top_k=5)
    context = " ".join(relevant_chunks)

    if mode == "concise":
        result = qa_model({"question": query, "context": context})
        return result["answer"]

    elif mode == "deep":
        prompt = f"Answer the question in detail using the context.\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
        output = deep_model(prompt, max_length=300, do_sample=False)[0]["generated_text"]
        return output.strip()

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="PDF Q/A Chatbot", layout="wide")
    st.title("📑 PDF Q/A Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        text = extract_text_from_pdf(uploaded_file)
        cleaned = clean_text(text)
        chunks = chunk_text(cleaned, chunk_size=500)
        embeddings = embed_chunks(chunks)

        st.success("✅ PDF processed! Ask your questions below.")

        mode = st.radio("Answer Mode:", ["Concise", "Deep"], horizontal=True)
        query = st.text_input("Enter your question:")

        if st.button("Get Answer"):
            if query:
                answer = generate_answer(query, chunks, embeddings, mode.lower())
                st.subheader("Answer:")
                st.write(answer)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
