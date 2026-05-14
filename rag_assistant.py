# ============================================================
# PROJECT 4: RAG-Based AI Research Assistant
# Stack: LangChain, ChromaDB, Ollama (free local LLM), Streamlit
# Install: pip install langchain langchain-community chromadb
#          streamlit pypdf sentence-transformers
# LLM:     Install Ollama from https://ollama.com → ollama pull llama3
# ============================================================

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType

CHROMA_DIR = "./chroma_db"
DOCS_DIR   = "./documents"   # put your PDFs here

# ---------- 1. Ingest & Embed Documents ----------
def build_vectorstore(docs_dir: str = DOCS_DIR) -> Chroma:
    """Load all PDFs from a folder, chunk, embed, and store."""
    loader = DirectoryLoader(docs_dir, glob="**/*.pdf",
                             loader_cls=PyPDFLoader)
    docs   = loader.load()
    print(f"Loaded {len(docs)} pages from {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 500,
        chunk_overlap = 50,
        separators    = ["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb   = Chroma.from_documents(
        chunks, embeddings,
        persist_directory=CHROMA_DIR
    )
    vectordb.persist()
    print(f"Vectorstore saved to {CHROMA_DIR}")
    return vectordb


def load_vectorstore() -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_DIR,
                  embedding_function=embeddings)


# ---------- 2. Build RAG Chain ----------
PROMPT_TEMPLATE = """
You are a helpful research assistant. Use the context below to answer
the question. If you don't find the answer in the context, say so clearly.
Always cite the source page numbers when available.

Context:
{context}

Question: {question}

Answer:"""

def build_rag_chain(vectordb: Chroma) -> RetrievalQA:
    llm     = Ollama(model="llama3", temperature=0.1)
    prompt  = PromptTemplate(
        template       = PROMPT_TEMPLATE,
        input_variables= ["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm                  = llm,
        chain_type           = "stuff",
        retriever            = vectordb.as_retriever(
            search_type      = "mmr",
            search_kwargs    = {"k": 5, "fetch_k": 10}
        ),
        return_source_documents = True,
        chain_type_kwargs    = {"prompt": prompt}
    )
    return chain


# ---------- 3. Agentic Layer ----------
def build_agent(rag_chain: RetrievalQA):
    llm = Ollama(model="llama3", temperature=0)

    def search_docs(query: str) -> str:
        result = rag_chain({"query": query})
        answer = result["result"]
        sources = set(
            doc.metadata.get("source", "unknown")
            for doc in result["source_documents"]
        )
        return f"{answer}\n\nSources: {', '.join(sources)}"

    tools = [
        Tool(
            name        = "Document Search",
            func        = search_docs,
            description = "Search uploaded research PDFs for specific information. "
                          "Use this for any question about the documents."
        )
    ]

    agent = initialize_agent(
        tools       = tools,
        llm         = llm,
        agent       = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose     = True,
        max_iterations = 5
    )
    return agent


# ---------- 4. Streamlit Chat UI ----------
def run_streamlit():
    import streamlit as st

    st.set_page_config(page_title="AI Research Assistant", layout="wide")
    st.title("AI Research Assistant (RAG)")
    st.caption("Upload PDFs → Chat with your documents")

    uploaded = st.sidebar.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True
    )

    if uploaded:
        Path(DOCS_DIR).mkdir(exist_ok=True)
        for file in uploaded:
            path = os.path.join(DOCS_DIR, file.name)
            with open(path, "wb") as f:
                f.write(file.read())
        with st.spinner("Processing and embedding documents..."):
            vectordb = build_vectorstore()
            rag_chain = build_rag_chain(vectordb)
            st.session_state["rag"] = rag_chain
        st.sidebar.success(f"{len(uploaded)} document(s) loaded!")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask anything about your documents..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        if "rag" not in st.session_state:
            st.warning("Please upload PDFs first.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching documents..."):
                    result  = st.session_state["rag"]({"query": question})
                    answer  = result["result"]
                    sources = result["source_documents"]
                    st.write(answer)
                    if sources:
                        with st.expander("Sources"):
                            for doc in sources[:3]:
                                st.write(f"- Page {doc.metadata.get('page','')} | {doc.metadata.get('source','')}")
                                st.caption(doc.page_content[:300] + "...")
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )


if __name__ == "__main__":
    import sys
    if "streamlit" in sys.argv[0]:
        run_streamlit()
    else:
        # CLI mode
        os.makedirs(DOCS_DIR, exist_ok=True)
        if not os.path.exists(CHROMA_DIR):
            vectordb = build_vectorstore()
        else:
            vectordb = load_vectorstore()
        chain = build_rag_chain(vectordb)
        while True:
            q = input("\nAsk a question (or 'quit'): ")
            if q.lower() == "quit":
                break
            result = chain({"query": q})
            print("\nAnswer:", result["result"])

# Run Streamlit: streamlit run rag_assistant.py
# Run CLI:       python rag_assistant.py
