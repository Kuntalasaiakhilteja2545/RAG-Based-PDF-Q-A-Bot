import streamlit as st
from qa_app import load_vectorstore, make_qa_chain

st.set_page_config(page_title="RAG PDF Q&A", layout="centered")
st.title("RAG PDF Q&A — Mini GenAI Project")

with st.sidebar:
    st.markdown("## Index / Management")
    if st.button("Reload index"):
        st.experimental_rerun()

try:
    vectordb = load_vectorstore()
    qa = make_qa_chain(vectordb)
except Exception as e:
    st.error(f"Could not load index: {e}")
    vectordb = None
    qa = None

prompt = st.text_area("Ask anything about your indexed PDFs:")
if st.button("Get answer") and prompt.strip():
    if qa is None:
        st.error("Index is not loaded. Run ingest.py to create the index first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            answer = qa.run(prompt)
        st.markdown("**Answer**")
        st.write(answer)

st.markdown("---")
st.markdown("**Indexed sources**")
if vectordb is not None:
    try:
        # best-effort display of sources
        docs = getattr(vectordb, 'docstore', None)
        st.write("Indexed documents available in vectorstore (metadata preview):")
        if docs:
            # display first 10 metadata entries
            meta_preview = []
            for i, (docid, doc) in enumerate(docs.items()):
                if i >= 10: break
                meta_preview.append({"docid": docid, "metadata": getattr(doc, 'metadata', None)})
            st.json(meta_preview)
        else:
            st.write("(No docstore metadata available)")
    except Exception:
        st.write("(Couldn't load source list)")
