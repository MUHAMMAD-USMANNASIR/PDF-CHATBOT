import os
os.environ["STREAMLIT_HOME"] = os.getcwd()  # fix permission issues in some hosts

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ----------------- Configuration ----------------- #
st.set_page_config(page_title="PDF Chatbot", layout="wide")
CHUNK_SIZE = 150      # words per chunk
CHUNK_OVERLAP = 50    # overlapping words between chunks
TOP_K = 5             # number of chunks to retrieve for context
QA_CONFIDENCE_THRESHOLD = 0.20  # below this, fallback to summarizer

# ----------------- Load models (cached) ----------------- #
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa = pipeline("question-answering", model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return embedder, qa, summarizer

embedder, qa_pipeline, summarizer = load_models()

# ----------------- PDF utilities ----------------- #
def extract_text_from_pdf(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
    except Exception:
        return ""
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts).strip()

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start += chunk_size - overlap
    return chunks

# ----------------- Indexing / retrieval ----------------- #
def build_faiss_index(chunks):
    if not chunks:
        return None, None
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    # ensure float32 and contiguity for faiss
    embeddings = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # use inner product on normalized vectors -> cosine similarity
    index.add(embeddings)
    return index, embeddings

def retrieve_top_k(query, index, chunks, k=TOP_K):
    if index is None or not chunks:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)
    k = min(k, len(chunks))
    distances, indices = index.search(q_emb, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        results.append({"index": int(idx), "score": float(dist), "text": chunks[idx]})
    return results

def build_context_from_results(results, max_chars=1200):
    context = ""
    for item in results:
        if len(context) + len(item["text"]) > max_chars:
            # include partial chunk if necessary
            remaining = max_chars - len(context)
            if remaining > 50:
                context += item["text"][:remaining]
            break
        context += item["text"] + "\n\n"
    return context.strip()

# ----------------- Streamlit UI ----------------- #
st.title("PDF Chatbot")
st.write("Upload a PDF and ask questions. The app extracts relevant passages and returns concise answers.")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

# initialize session state containers
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, answer, score)

if uploaded_file:
    if st.button("Process PDF", key="process_pdf"):
        with st.spinner("Extracting text and building index..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("No extractable text found. If the PDF contains scanned images, OCR is required.")
            else:
                chunks = split_text_into_chunks(text)
                if not chunks:
                    st.error("Failed to split document into chunks.")
                else:
                    index, embeddings = build_faiss_index(chunks)
                    if index is None:
                        st.error("Failed to build the search index.")
                    else:
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.embeddings = embeddings
                        st.success(f"Document processed: {len(chunks)} chunks indexed.")

# Query UI only shown after index exists
if st.session_state.index is not None and st.session_state.chunks:
    st.subheader("Ask a question about the document")
    query = st.text_input("Enter your question", key="user_query_input")

    if query:
        with st.spinner("Retrieving answer..."):
            top_results = retrieve_top_k(query, st.session_state.index, st.session_state.chunks, k=TOP_K)
            if not top_results:
                st.error("No relevant passages found.")
            else:
                # build a context from top results (limit characters to keep QA model in-range)
                context = build_context_from_results(top_results, max_chars=1500)
                try:
                    qa_input = {"question": query, "context": context}
                    qa_output = qa_pipeline(qa_input)
                    answer = qa_output.get("answer", "").strip()
                    score = float(qa_output.get("score", 0.0))
                except Exception:
                    answer = ""
                    score = 0.0

                # fallback: if QA low-confidence or empty, return a short summary of the retrieved chunks
                if (not answer) or (score < QA_CONFIDENCE_THRESHOLD):
                    # combine top chunk texts for summarization input
                    combined = " ".join([r["text"] for r in top_results])
                    # limit size to summarizer capacity
                    combined = combined[:4000]
                    try:
                        summary = summarizer(combined, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
                        result_text = f"{summary}\n\n(Confidence: low — produced by summarization fallback.)"
                    except Exception:
                        result_text = "Could not produce an answer with high confidence."
                    st.write("Answer:")
                    st.write(result_text)
                    st.session_state.history.append((query, result_text, score))
                else:
                    st.write("Answer:")
                    st.write(answer)
                    st.write(f"Confidence: {score:.2f}")
                    st.session_state.history.append((query, answer, score))

                # show sources (top matches) in expanders for traceability
                with st.expander("Show retrieved passages and similarity scores"):
                    for item in top_results:
                        st.markdown(f"**Chunk #{item['index']} — score: {item['score']:.3f}**")
                        st.write(item["text"])

# Show conversation history
if st.session_state.history:
    st.subheader("Query history")
    for i, (q, a, s) in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.markdown("---")
