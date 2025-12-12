import os
import pickle
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)

def create_faiss_index(pdf_paths: list, persist_path: str = VECTORSTORE_DIR):
    texts = []
    metadatas = []
    for p in pdf_paths:
        txt = load_pdf_text(p)
        chunks = chunk_text(txt)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source": os.path.basename(p), "chunk": i})

    embed_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = FAISS.from_texts(texts, embed_model, metadatas=metadatas)
    os.makedirs(persist_path, exist_ok=True)
    faiss_path = os.path.join(persist_path, "faiss_index.pkl")
    with open(faiss_path, "wb") as f:
        pickle.dump(vectordb, f)
    print("Saved FAISS index to", faiss_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ingest.py path/to/doc1.pdf [path/to/doc2.pdf ...]")
        sys.exit(1)
    pdfs = sys.argv[1:]
    create_faiss_index(pdfs)
