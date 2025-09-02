import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# PDF LOADING & CHUNKING
# -----------------------------
def load_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


# -----------------------------
# EMBEDDINGS & SEARCH
# -----------------------------
def embed_chunks(chunks, model):
    return model.encode(chunks)


def search_chunks(query, chunks, embeddings, model, top_k=5):
    # Encode query
    query_emb = model.encode([query])

    # Similarity
    sims = cosine_similarity(query_emb, embeddings)[0]

    # Top-k
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]


# -----------------------------
# ANSWER GENERATION
# -----------------------------
def generate_answer(query, chunks, embeddings, model, mode="concise"):
    relevant_chunks = search_chunks(query, chunks, embeddings, model, top_k=3)
    context = " ".join(relevant_chunks)

    if mode == "concise":
        generator = pipeline("text-generation", model="distilgpt2")
        prompt = f"Answer briefly: {query}\nContext: {context}\nAnswer:"
        return generator(prompt, max_length=100, num_return_sequences=1)[0]["generated_text"]

    elif mode == "deep":
        generator = pipeline("text2text-generation", model="google/flan-t5-base")
        prompt = f"Answer in detail using the context:\n{context}\nQuestion: {query}\nAnswer:"
        return generator(prompt, max_length=300, num_return_sequences=1)[0]["generated_text"]

    else:
        return "Invalid mode selected."


# -----------------------------
# STREAMLIT APP
# -----------------------------
def main():
    st.title("📘 PDF Q&A Chatbot")
    st.write("Upload a PDF and ask questions from it!")

    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    if uploaded_file:
        text = load_pdf(uploaded_file)
        st.success("PDF loaded successfully!")

        chunks = chunk_text(text)
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embed_chunks(chunks, model)

        query = st.text_input("Ask a question about the PDF:")
        mode = st.radio("Select answer mode:", ["concise", "deep"])

        if st.button("Get Answer") and query:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, chunks, embeddings, model, mode.lower())
                st.write("### Answer:")
                st.write(answer)


if __name__ == "__main__":
    main()
