import os
os.environ["STREAMLIT_HOME"] = os.getcwd()  # avoid permission issues on some hosts

import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
from transformers import pipeline

# ---------------- Configuration ---------------- #
st.set_page_config(page_title="PDF Chatbot", layout="wide")
CHUNK_SIZE = 200            # words per chunk
CHUNK_OVERLAP = 50          # overlap words between chunks
CANDIDATE_K = 10            # candidates retrieved from vector search
RERANK_TOPK = 5             # top candidates re-ranked by cross-encoder
FINAL_CONTEXT_TOPK = 3      # chunks used to build context for QA
QA_CONF_THRESHOLD = 0.35    # QA confidence threshold (0..1)
SIMILARITY_THRESHOLD = 0.10 # minimum inner-product (cosine) similarity for candidates

# ---------------- Load models (cached) ---------------- #
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # Cross-encoder for re-ranking (lightweight and effective)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # QA model (extractive)
    qa_pipeline = pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2",
        tokenizer="deepset/roberta-base-squad2",
        device=-1
    )
    # Summarizer fallback (smaller distil variant for speed)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
    return embedder, cross_encoder, qa_pipeline, summarizer

embedder, cross_encoder, qa_pipeline, summarizer = load_models()

# ---------------- PDF utilities ---------------- #
def extract_text_from_pdf(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
    except Exception:
        return ""
    texts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            texts.append(page_text)
    return "\n".join(texts).strip()

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

# ---------------- Indexing & retrieval ---------------- #
def build_faiss_index(chunks):
    if not chunks:
        return None, None
    # embeddings: (N, dim)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors -> cosine similarity
    index.add(embeddings)
    return index, embeddings

def retrieve_candidates(query, index, chunks, k=CANDIDATE_K, similarity_threshold=SIMILARITY_THRESHOLD):
    if index is None or not chunks:
        return []
    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype=np.float32)
    faiss.normalize_L2(q_emb)
    k = min(k, len(chunks))
    D, I = index.search(q_emb, k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        # Only include candidates above similarity threshold
        if score >= similarity_threshold:
            results.append({"index": int(idx), "score": float(score), "text": chunks[idx]})
    return results

def rerank_with_cross_encoder(query, candidates, topk=RERANK_TOPK):
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    try:
        scores = cross_encoder.predict(pairs)
    except Exception:
        # fallback: return original order
        return candidates[:topk]
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:min(topk, len(candidates))]

def build_context_from_candidates(candidates, max_chars=1500):
    context = ""
    for item in candidates:
        add_text = item["text"]
        if len(context) + len(add_text) > max_chars:
            remaining = max_chars - len(context)
            if remaining > 50:
                context += add_text[:remaining]
            break
        context += add_text + "\n\n"
    return context.strip()

# ---------------- Answering ---------------- #
def answer_query(query, index, chunks):
    candidates = retrieve_candidates(query, index, chunks, k=CANDIDATE_K)
    if not candidates:
        return {"answer": "No relevant passages found in the document.", "confidence": 0.0, "sources": []}

    reranked = rerank_with_cross_encoder(query, candidates, topk=RERANK_TOPK)
    # build context from top reranked passages
    context_candidates = reranked[:FINAL_CONTEXT_TOPK]
    context = build_context_from_candidates(context_candidates, max_chars=1600)

    if not context:
        return {"answer": "Unable to build context for the question.", "confidence": 0.0, "sources": candidates}

    # run QA extractor
    try:
        qa_input = {"question": query, "context": context}
        qa_out = qa_pipeline(qa_input)
        answer_text = qa_out.get("answer", "").strip()
        score = float(qa_out.get("score", 0.0))
    except Exception:
        answer_text = ""
        score = 0.0

    # if QA confidence low or answer empty, fallback to summarization of top candidates
    if (not answer_text) or (score < QA_CONF_THRESHOLD):
        combined = " ".join([c["text"] for c in reranked])
        combined = combined[:4000]  # limit for summarizer
        try:
            summary = summarizer(combined, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
            return {"answer": summary, "confidence": score, "sources": reranked}
        except Exception:
            return {"answer": "Could not produce a confident answer.", "confidence": score, "sources": reranked}

    return {"answer": answer_text, "confidence": score, "sources": reranked}

# ---------------- Streamlit app ---------------- #
st.title("PDF Chatbot")
st.write("Upload a PDF document, process it, then ask questions. Answers include confidence and sources.")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"], key="pdf_upload")

# session state for persistent index/chunks/history
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file:
    if st.button("Process PDF", key="process_pdf"):
        with st.spinner("Extracting and indexing..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error("No extractable text found in this PDF. If the file is scanned or image-based, enable OCR first.")
            else:
                chunks = split_text_into_chunks(text)
                if not chunks:
                    st.error("Document splitting failed or resulted in no chunks.")
                else:
                    index, embeddings = build_faiss_index(chunks)
                    if index is None:
                        st.error("Index building failed.")
                    else:
                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.embeddings = embeddings
                        st.success(f"Document processed: {len(chunks)} chunks indexed.")

# show query UI only when index exists
if st.session_state.index is not None and st.session_state.chunks:
    st.subheader("Ask a question about the document")
    query = st.text_input("Enter your question", key="user_question_input")

    if query:
        with st.spinner("Finding answer..."):
            result = answer_query(query, st.session_state.index, st.session_state.chunks)

        st.markdown("**Answer:**")
        st.write(result["answer"])
        st.markdown(f"**Confidence:** {result['confidence']:.2f}")

        # append history
        st.session_state.history.append({"question": query, "answer": result["answer"], "confidence": result["confidence"]})

        # show sources for transparency
        with st.expander("Retrieved passages (sources)"):
            for src in result.get("sources", []):
                idx = src.get("index", "?")
                sc = src.get("rerank_score", src.get("score", 0.0))
                st.markdown(f"**Chunk #{idx} — score: {sc:.4f}**")
                st.write(src.get("text", "")[:2000])  # truncate long passages in UI

# show query history
if st.session_state.history:
    st.subheader("Query history")
    for item in reversed(st.session_state.history[-20:]):
        st.markdown(f"**Q:** {item['question']}")
        st.markdown(f"**A:** {item['answer']}")
        st.markdown(f"**Confidence:** {item['confidence']:.2f}")
        st.markdown("---")
