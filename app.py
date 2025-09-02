import streamlit as st
import faiss
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------
# Load models
# ---------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generative QA model
gen_model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
generator = pipeline("text2text-generation", model=gen_model, tokenizer=tokenizer)

# ---------------------
# Functions
# ---------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, chunk_size=400, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def generate_answer(query, chunks, index, embeddings, top_k=3):
    """Retrieve top passages and generate an answer"""
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(np.array(query_emb, dtype="float32"), top_k)

    retrieved_passages = [chunks[i] for i in I[0] if i < len(chunks)]
    context = " ".join(retrieved_passages)

    if not context.strip():
        return "Sorry, I couldn’t find anything relevant."

    # Build prompt
    prompt = f"Answer the question based on the document:\n\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    output = generator(prompt, max_length=200, num_return_sequences=1, do_sample=False)[0]["generated_text"]
    return output.strip()

# ---------------------
# Streamlit App
# ---------------------
st.title("PDF Chatbot")
st.write("Ask questions about your uploaded PDF with more confident, deep answers.")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, embeddings = build_faiss_index(chunks)

    st.success("PDF processed. You can now ask questions.")

    query = st.text_input("Enter your question:")
    if query:
        answer = generate_answer(query, chunks, index, embeddings)
        st.subheader("Answer:")
        st.write(answer)
