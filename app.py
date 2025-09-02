import streamlit as st
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------
# Load Models Once (cached)
# -------------------------
@st.cache_resource
def load_models():
    embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    deep_generator = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    precise_generator = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return embeddings, deep_generator, precise_generator

embeddings, deep_generator, precise_generator = load_models()

# -------------------------
# PDF Processing
# -------------------------
def load_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# -------------------------
# Embedding + FAISS
# -------------------------
def create_faiss_index(chunks, embeddings):
    vectors = embeddings.encode(chunks)
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors, dtype="float32"))
    return index, vectors

def retrieve_relevant_chunks(query, chunks, index, embeddings, k=5):
    query_vec = embeddings.encode([query])
    distances, indices = index.search(np.array(query_vec, dtype="float32"), k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

# -------------------------
# Answer Generation
# -------------------------
def generate_answer(query, chunks, index, embeddings, mode="deep"):
    retrieved_chunks = retrieve_relevant_chunks(query, chunks, index, embeddings)
    context = " ".join(retrieved_chunks)[:2000]  # keep short for accuracy

    if not context.strip():
        return "No relevant information found in the document."

    if mode == "deep":
        prompt = f"Answer the question based only on the following text:\n\n{context}\n\nQuestion: {query}\nAnswer:"
        output = deep_generator(prompt, max_length=300, num_return_sequences=1, do_sample=False)[0]["generated_text"]
        return output.strip()

    elif mode == "precise":
        result = precise_generator(question=query, context=context)
        return result["answer"].strip()

    else:
        return "Invalid mode selected. Choose 'deep' or 'precise'."

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("PDF Q&A System")
    st.write("Upload a PDF and ask questions. Choose between Deep or Precise answers.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file is not None:
        text = load_pdf(uploaded_file)
        chunks = chunk_text(text)
        index, vectors = create_faiss_index(chunks, embeddings)

        st.success("PDF processed successfully! You can now ask questions.")

        mode = st.radio("Select Answer Mode:", ["deep", "precise"], horizontal=True)

        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, chunks, index, embeddings, mode=mode)
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()
