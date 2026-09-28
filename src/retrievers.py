from pathlib import Path
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_cohere import CohereRerank
load_dotenv()

EMB = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
CHROMA_DIR = 'chroma'
_CROSS_ENCODER = None

def get_store(docs=None): 
    if Path(CHROMA_DIR).exists():
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMB)
    return Chroma.from_documents(docs, embedding=EMB, persist_directory=CHROMA_DIR)

def load_store():
    """Open an already-built index. No docs, no rebuilt"""
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMB)

def get_hybrid(docs, store, k_each: int=10):
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k_each
    dense = store.as_retriever(search_kwargs= { "k": k_each})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.4, 0.6])

def get_reranked(store, k_wide: int = 30, top_n : int = 5):
    return ContextualCompressionRetriever(
        base_compressor=CohereRerank(model='rerank-v3.5', top_n=top_n),
        base_retriever=store.as_retriever(search_kwargs={"k": k_wide})
    )