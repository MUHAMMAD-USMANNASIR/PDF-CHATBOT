import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np

# ----------------------------
# 1. Load embedding model
# ----------------------------
@st.cache_resource
def load_embeddings():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# 2. Extract text from PDF
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text") + " "
    return text.strip()

# ----------------------------
# 3. Chunk text
# ----------------------------
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# ----------------------------
# 4. Build embeddings
# ----------------------------
def build_embeddings(chunks, model):
    vectors = model.encode(chunks, convert_to_tensor=True).cpu().numpy()
    return vectors

# ----------------------------
# 5. Search relevant chunks
# ----------------------------
def search_chunks(query, chunks, embeddings, model, top_k=3):
    query_vec = model.encode([query], convert_to_tensor=True).cpu().numpy()[0]
    scores = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )
    top_ids = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_ids]

# ----------------------------
# 6. Generate answer (retrieval only)
# ----------------------------
def generate_answer(query, chunks, embeddings, embed_model, mode="concise"):
    relevant_chunks = search_chunks(query, chunks, embeddings, embed_model, top_k=3)
    
    if mode == "concise":
        return f"Answer (from PDF): {relevant_chunks[0]}"
    
    elif mode == "deep":
        return "Answer (from PDF):\n\n" + "\n\n".join(relevant_chunks)
    
    else:
        return "❌ Invalid mode."

# ----------------------------
# 7. Streamlit UI
# ----------------------------
def main():
    st.title("📄 PDF Q&A Bot (Retriever Only)")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        with st.spinner("Extracting text..."):
            pdf_text = extract_text_from_pdf(uploaded_file)

        chunks = chunk_text(pdf_text, chunk_size=500)
        embed_model = load_embeddings()

        with st.spinner("Building embeddings..."):
            embeddings = build_embeddings(chunks, embed_model)

        st.success("✅ PDF processed successfully!")

        query = st.text_input("Ask a question about your PDF:")
        mode = st.radio("Answer Mode:", ["concise", "deep"])

        if query:
            with st.spinner("Searching..."):
                answer = generate_answer(query, chunks, embeddings, embed_model, mode)
            st.write(answer)

if __name__ == "__main__":
    main()
