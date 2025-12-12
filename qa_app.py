import os
import pickle
from dotenv import load_dotenv
from langchain.llms import OpenAI
load_dotenv()

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./vectorstore")

def load_vectorstore(path: str = VECTORSTORE_DIR):
    faiss_path = os.path.join(path, "faiss_index.pkl")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}. Run ingest.py first.")
    with open(faiss_path, "rb") as f:
        vectordb = pickle.load(f)
    return vectordb

def make_qa_chain(vectordb):
    llm = OpenAI(temperature=0)
    retriever = vectordb.as_retriever(search_type="hybrid", search_kwargs={"k": 4})
    from langchain.chains import RetrievalQA
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa

if __name__ == "__main__":
    vs = load_vectorstore()
    qa = make_qa_chain(vs)
    while True:
        q = input("Question> ")
        if not q.strip():
            break
        print(qa.run(q))
