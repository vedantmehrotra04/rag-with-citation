from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

PROMPT = """You are answering questions about FastAPI using only the documentation excerpts below.

Rules:
- Use ONLY the excerpts. Do not use prior knowledge.
- After each claim, cite the excerpt number in square brackets, like [2].
- If the excerpts do not contain the answer, reply with exactly: I don't know
- Be concise. Three sentences maximum.

Excerpts:
{context}

Question: {question}

Answer:"""

_LLM = None

def get_llm():
    global _LLM
    if _LLM is None:
        _LLM = ChatGoogleGenerativeAI(model='gemini-3.6-flash', temperature=0)
    return _LLM

def format_context(docs: list[Document]) -> str:
    text_chunks = []
    for i, d in enumerate(docs, start=1):
        heading = d.metadata.get("heading_path") or d.metadata.get("title", "")
        text_chunks.append(f"[{i}] {heading}\n{d.metadata['original_text']}")
    return "\n\n".join(text_chunks)

def as_text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(
        b.get("text", "") for b in content if isinstance(b,dict)
    )

def answer(question: str, retriever, llm=None) -> dict:
    llm = llm or get_llm()
    docs = retriever.invoke(question)

    text = as_text(llm.invoke(
        PROMPT.format(context=format_context(docs), question=question)
    ))

    return {
        "answer": text,
        "refused": text.strip().lower().startswith("i don't know"),
        "sources": [
            {
                "n": i,
                "heading": d.metadata.get("heading_path", ""),
                "url": d.metadata["url"]
            }
            for i, d in enumerate(docs, start=1)
        ]
    }
