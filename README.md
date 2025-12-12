RAG-Based PDF Q&A Bot

Retrieval-Augmented Generation (RAG) over PDFs — a compact, production-oriented mini project that demonstrates how to index PDF documents, build embeddings, and serve context-aware Q&A using LangChain and a vector store.

This repo contains:

PDF ingestion and chunking

Embedding-based indexing (FAISS by default)

A retrieval + LLM answer pipeline (RAG)

A simple Streamlit demo UI

Dockerfile, .env.example, and tips for deployment

Features

Load and parse PDFs (supports many PDFs; OCR not included by default)

Chunk documents with overlap to preserve context

Create dense embeddings (HuggingFace or OpenAI) and store in FAISS

Retrieve relevant chunks and generate answers with an LLM

Streamlit UI for interactive testing

Easy to swap vectorstores or embedding providers

p -d rag-pdf-qa

Create a .env file from the example:

cp .env.example .env
# open .env and add OPENAI_API_KEY if using OpenAI embeddings/LLM

Install dependencies (create a virtualenv if you prefer):

pip install -r requirements.txt

Place PDFs in sample_data/ or pass PDFs to the ingestion script.

Build the vector index:

python ingest.py sample_data/your_file.pdf
# This creates ./vectorstore/faiss_index.pkl by default

Run the Streamlit demo:

streamlit run app_streamlit.py

Or use the CLI QA:

python qa_app.py

Files & purpose

ingest.py — load PDFs, chunk text, compute embeddings, and build a FAISS index (pickled to vectorstore/)

qa_app.py — load the vectorstore and run a RetrievalQA loop (uses OpenAI LLM by default)

app_streamlit.py — small Streamlit UI to query the indexed PDFs

utils.py — tiny helpers (e.g., list PDFs)

requirements.txt — Python packages used

Dockerfile — simple container image that runs the Streamlit app

.env.example — example environment variables

LICENSE — MIT license template

Environment variables

Copy .env.example -> .env and configure:

OPENAI_API_KEY=sk-...

EMBEDDING_MODEL=all-MiniLM-L6-v2   # if using SentenceTransformers

VECTORSTORE_DIR=./vectorstore

CHUNK_SIZE=1000

CHUNK_OVERLAP=200


If you use OpenAIEmbeddings or OpenAI LLMs, set OPENAI_API_KEY.

CHUNK_SIZE and CHUNK_OVERLAP tune retrieval granularity.

How it works (high-level)

PDF ingestion: ingest.py reads PDF pages, concatenates text, and splits into overlapping chunks using RecursiveCharacterTextSplitter.

Embeddings: Each chunk is converted into a vector embedding (HuggingFace or OpenAI).

Vector store: Embeddings and metadata are stored in FAISS (pickled). For production consider Chroma, Pinecone, or Weaviate with persistence.

Retrieval + Generation: On a query, the vectorstore retrieves the top-k chunks as context. The LLM generates an answer conditioned on those chunks (RAG).

UI / CLI: Streamlit or CLI wraps the pipeline for testing.

Common customizations

Switch embeddings to OpenAI: Replace HuggingFaceEmbeddings with OpenAIEmbeddings in ingest.py and qa_app.py. Ensure OPENAI_API_KEY is set.

Use a different vector DB: Swap FAISS for Chroma/Pinecone/Weaviate via LangChain's VectorStore wrappers — update persistence code.

Local LLMs: Swap OpenAI LLM class with a local LLM wrapper (e.g., HuggingFaceHub, TextGenerationInference, or a custom adapter).

Add OCR: Integrate Tesseract or AWS Textract for scanned documents before ingestion.

Production & deployment tips

Use persistent vector DBs (Pinecone, Chroma with persistent storage, Weaviate) for multi-user apps.

Add an API layer (FastAPI) between the UI and RAG pipeline for scalability and authentication.

Rate-limit LLM calls and cache embeddings.

Secure the .env and rotate API keys.

Add tests for ingestion, vectorstore creation, and retrieval correctness.

Security & cost

OpenAI calls incur cost; carefully limit context and k.

Sanitize input if exposing publicly.

Don’t commit .env or API keys to Git.

Troubleshooting

No text extracted from PDF: The PDF might be scanned images — use OCR (Tesseract) to extract text.

FAISS loading errors: Confirm vectorstore/faiss_index.pkl exists. The pickle includes the vectorstore object — if you change the LangChain / embeddings library versions, re-ingest.

Slow queries: Reduce k or use a faster retriever; use more compact embeddings (e.g., all-MiniLM-L6-v2).

Model errors: Ensure the LLM class used in qa_app.py matches your available model and API keys.

Example usage snippet (Python)
from qa_app import load_vectorstore, make_qa_chain

vs = load_vectorstore()
qa = make_qa_chain(vs)
print(qa.run("What is the main idea in the document?"))

Next steps / ideas (great for README to attract attention)

Add page-level and chunk-level source attribution in answers (show which PDF and page/chunk the answer came from).

Add OCR pipeline for scanned docs (Tesseract).

Add user auth and access control for private documents.

Add incremental ingestion with embedding caching.

Add GitHub Actions to lint and run tests on push.

Contributing

Contributions welcome — open issues or PRs for bug fixes, new features, or improved UI.

Suggested checklist for PR:

Add tests for ingestion and retrieval

Update CHANGELOG.md for significant changes

Ensure secrets are not committed
