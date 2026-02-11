import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import chromadb
import numpy as np
from unstructured.partition.auto import partition
import spacy
import re


# ===================== CONFIG =====================

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

CHROMA_PATH = "./hf_vector_db"
COLLECTION_NAME = "my_document"

client = InferenceClient(token=HF_TOKEN)

st.set_page_config(page_title="Local RAG Demo", page_icon="Robot")
st.title("RAG Engine with Qwen3-4B (Streamlit Demo)")

if not HF_TOKEN:
    st.error("HF_TOKEN not found! Create a .env file with: HF_TOKEN=your_token_here")
    st.stop()


# ========= helper: clean metadata for Chroma (no None values) =========

def clean_metadata(meta: dict) -> dict:
    cleaned = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


# ========= sentence-aware / hybrid chunking helpers =========

@st.cache_resource
def load_spacy_model():
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


def advanced_chunking(text: str, max_chars: int = 1000, overlap_chars: int = 200) -> list[str]:
    text = text.replace("\r\n", "\n")
    
    raw_paragraphs = re.split(r'\n\s*\n', text)
    
    paragraphs = []
    for p in raw_paragraphs:
        p_clean = re.sub(r'[ \t]+', ' ', p).strip()
        p_clean = p_clean.replace("\n", " ")
        if p_clean:
            paragraphs.append(p_clean)
            
    chunks = []
    current_chunk_paragraphs = []
    current_chunk_len = 0
    
    nlp = load_spacy_model()

    for para in paragraphs:
        para_len = len(para)
        
        if current_chunk_len + para_len + 2 > max_chars:
            if current_chunk_paragraphs:
                chunks.append("\n\n".join(current_chunk_paragraphs))
                
                overlap_buffer = []
                overlap_len = 0
                for prev_para in reversed(current_chunk_paragraphs):
                    if overlap_len + len(prev_para) + 2 <= overlap_chars:
                        overlap_buffer.insert(0, prev_para)
                        overlap_len += len(prev_para) + 2
                    else:
                        break
                
                current_chunk_paragraphs = overlap_buffer
                current_chunk_len = overlap_len
            
            if len(para) > max_chars:
                doc = nlp(para)
                sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
                
                current_sent_chunk = []
                current_sent_len = 0
                
                for sent in sentences:
                    if current_sent_len + len(sent) + 1 > max_chars:
                         if current_sent_chunk:
                             chunks.append(" ".join(current_sent_chunk))
                             current_sent_chunk = []
                             current_sent_len = 0
                    
                    current_sent_chunk.append(sent)
                    current_sent_len += len(sent) + 1
                
                if current_sent_chunk:
                    chunks.append(" ".join(current_sent_chunk))
            else:
                current_chunk_paragraphs.append(para)
                current_chunk_len += para_len + 2
        
        else:
            current_chunk_paragraphs.append(para)
            current_chunk_len += para_len + 2
            
    if current_chunk_paragraphs:
        chunks.append("\n\n".join(current_chunk_paragraphs))
        
    return chunks


# ===================== SIDEBAR UI =====================

with st.sidebar:
    st.header("1. Configuration")
    st.write(f"Embedding model: `{EMBEDDING_MODEL}`")
    st.write(f"LLM model: `{LLM_MODEL}`")
    st.success("HF_TOKEN loaded from .env")

    st.header("2. Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a Doc (PDF/Docx/TXT)", type=["docx", "pdf", "txt"]
    )

    process_btn = st.button("Process Document")

st.markdown("---")


# ===================== VECTOR DB BUILDING =====================

def build_vector_db_from_path(doc_path: str, display_name: str):
    st.info(f"Loading and processing document: **{display_name}**")

    if not Path(doc_path).exists():
        st.error(f"ERROR: {display_name} not found at path {doc_path}")
        st.stop()

    texts = []
    metadatas = []
    ids = []
    
    # === UNIFIED TEXT EXTRACTION USING UNSTRUCTURED FOR ALL FILES ===
    st.info("Extracting text using unstructured.partition()...")
    elements = partition(filename=doc_path)
    current_text = "\n\n".join([str(el) for el in elements if str(el).strip()])
    
    raw_chunks = advanced_chunking(current_text, max_chars=1000)
    
    for i, chunk_text in enumerate(raw_chunks):
        texts.append(chunk_text)
        meta = {"source": display_name, "chunk_index": i}
        metadatas.append(clean_metadata(meta))
        ids.append(f"chunk_{i:05d}")
            
    st.write(f"Extracted {len(texts)} chunks.")

    # Create / reset Chroma collection
    st.info("Creating/updating vector database...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    if COLLECTION_NAME in [c.name for c in chroma_client.list_collections()]:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except Exception as e:
            st.error(f"Error resetting DB: {e}")

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and store
    st.info("Generating embeddings and storing...")
    if not texts:
        st.warning("No text extracted!")
        return None

    total = len(texts)
    progress_bar = st.progress(0)
    batch_size = 32
    
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        batch_texts_for_embed = [f"passage: {t}" for t in batch_texts]

        try:
            response = client.feature_extraction(batch_texts_for_embed, model=EMBEDDING_MODEL)
            embeddings = [np.array(vec).flatten().tolist() for vec in response]
            
            collection.add(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_metas,
            )
        except Exception as e:
            st.error(f"Embedding error batch {i}: {e}")
            continue

        progress_bar.progress(min((i + batch_size) / total, 1.0))

    db = chroma_client.get_collection(COLLECTION_NAME)
    st.success(f"Vector DB ready! Total chunks: {collection.count()}")
    return db


# ===================== UNIVERSAL MULTI-QUERY FUNCTION (ONLY ADDITION) =====================
def universal_multi_query(question: str):
    variants = [
        question,
        f"Answer this question: {question}",
        f"Explain: {question}",
        f"What is said about: {question}",
        f"Tell me about: {question}",
        f"Find information regarding: {question}",
    ]
    
    all_chunks = []
    all_ids = []
    seen_ids = set()

    for q in variants:
        try:
            emb = np.array(client.feature_extraction(f"query: {q}", model=EMBEDDING_MODEL)).flatten().tolist()
            results = db.query(query_embeddings=[emb], n_results=8)
            
            for doc, doc_id in zip(results["documents"][0], results["ids"][0]):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_chunks.append(doc)
                    all_ids.append(doc_id)
        except:
            continue
    
    return all_chunks[:10], all_ids[:10]


# ===================== HANDLE DOCUMENT PROCESSING =====================

if process_btn:
    if uploaded_file is None:
        st.warning("Please upload a document first.")
        st.stop()

    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    db = build_vector_db_from_path(tmp_path, uploaded_file.name)
    st.session_state["db"] = db
    st.session_state["doc_name"] = uploaded_file.name


# ===================== CHAT UI =====================

st.subheader("Chat with your document")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_q = st.chat_input("Ask anything about the processed document...")

if user_q:
    if "db" not in st.session_state:
        st.warning("First upload & process a document in the sidebar.")
    else:
        db = st.session_state["db"]

        st.session_state["messages"].append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # === ONLY CHANGE: Use universal_multi_query instead of single query ===
                docs, ids = universal_multi_query(user_q)

                context_str = ""
                for idx, (doc_text, cid) in enumerate(zip(docs, ids), start=1):
                    context_str += f"\n[SOURCE {idx}]:\n{doc_text}\n"

                system_prompt = (
                    "You are a precise document-answering assistant.\n"
                    "You have been provided with chunks of text from a document, labeled [SOURCE X].\n"
                    "YOU ARE FORBIDDEN FROM:\n"
                    "• Making up any information\n"
                    "• Guessing\n"
                    "• Using knowledge outside these sources\n"
                    "• Paraphrasing loosely\n\n"
                    "YOU MUST:\n"
                    "1. Answer ONLY if the exact information is written in the sources\n"
                    "2. Quote numbers, names, dates, formulas exactly as written\n"
                    "3. Cite every single fact with [1], [2], etc. at the end of the sentence\n"
                    "4. If the answer is not 100% present → reply EXACTLY:\n"
                    "5. Quote values EXACTLY as written"
                    "6. If multiple sources support a sentence, cite all."
                    "   'The document does not contain this information.'\n\n"
                    "Never say 'according to the document' or 'it appears'. Just give the facts."

                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_q}\n\nAnswer:"}
                ]

                try:
                    response = client.chat_completion(
                        messages=messages, 
                        model=LLM_MODEL, 
                        max_tokens=600, 
                        temperature=0.1
                    )
                    answer = response.choices[0].message.content.strip()
                except Exception as e:
                    answer = f"Error generating response: {e}"

                st.markdown(answer)
                st.session_state["messages"].append({"role": "assistant", "content": answer})

                with st.expander("View Source Chunks"):
                    for i, (doc, id_) in enumerate(zip(docs, ids), start=1):
                        st.markdown(f"**[SOURCE {i}]** (ID: {id_})")
                        st.text(doc)
                        st.divider()