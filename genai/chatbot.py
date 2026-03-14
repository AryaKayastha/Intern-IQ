"""
genai/chatbot.py
----------------
LangChain RAG chatbot chain connecting ChromaDB → Ollama (Llama3).
Provides a simple ask(question) function used by Streamlit Tab 7.

Falls back gracefully if Ollama is unavailable (returns context only).
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, os.getenv("CHROMA_PERSIST_DIR", "genai/chroma_store"))
OLLAMA_HOST  = os.getenv("OLLAMA_HOST",  "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

_retriever = None
_chain     = None


def _build_retriever():
    """Lazy-load ChromaDB retriever."""
    import chromadb
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings  import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    client     = chromadb.PersistentClient(path=CHROMA_DIR)
    vectorstore = Chroma(
        client              = client,
        collection_name     = "intern_iq",
        embedding_function  = embeddings,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def _build_llm_and_prompt():
    """Lazy-build Ollama LLM and PromptTemplate."""
    try:
        from langchain_community.llms    import Ollama
        from langchain_core.prompts      import PromptTemplate

        template = """You are an intelligent analytics assistant for the InternIQ platform.
Use the following data context to answer the question accurately and concisely.
If the answer is not in the context, say "I don't have enough data to answer that."

Context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template       = template,
            input_variables=["context", "question"],
        )

        llm = Ollama(base_url=OLLAMA_HOST, model=OLLAMA_MODEL, temperature=0.1)
        return {"llm": llm, "prompt": prompt}, "ollama"
    except Exception as e:
        print(f"[WARN] Ollama not available: {e}. Chatbot will return context only.")
        return None, "context_only"


def _get_components():
    global _retriever, _chain
    if _retriever is None:
        _retriever = _build_retriever()
    if _chain is None:
        _chain = _build_llm_and_prompt()
    return _retriever, _chain


def ask(question: str) -> dict:
    """
    Ask a natural language question about intern data.

    Returns:
        {
            "answer":  str,
            "sources": list[str],   # source chunk IDs / text snippets
            "mode":    "ollama" | "context_only" | "unavailable"
        }
    """
    if not os.path.exists(CHROMA_DIR):
        return {
            "answer": "Vector store not found. Please run `python -m genai.embeddings` first.",
            "sources": [],
            "mode": "unavailable",
        }

    try:
        retriever, chain_tuple = _get_components()
        components, mode = chain_tuple

        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        sources = [d.metadata.get("type", "") + ": " + d.page_content[:100] for d in docs]

        if mode == "context_only" or components is None:
            return {
                "answer": f"[Context-only mode — Ollama unavailable]\n\n{context}",
                "sources": sources,
                "mode": "context_only",
            }

        llm = components["llm"]
        prompt = components["prompt"]
        formatted_prompt = prompt.format(context=context, question=question)
        
        answer = llm.invoke(formatted_prompt)
        
        return {"answer": answer, "sources": sources, "mode": "ollama"}

    except Exception as e:
        return {"answer": f"Error: {e}", "sources": [], "mode": "error"}
