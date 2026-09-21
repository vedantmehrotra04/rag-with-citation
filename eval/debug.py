import json
from src.ingest import load_and_chunk
from src.retrievers import get_store

docs = load_and_chunk()
questions = json.load(open("eval/questions.json"))
store = get_store(docs)

# 1 — is anything actually in the store?
print("chunks built:", len(docs))
print("in chroma   :", store._collection.count())

# 2 — does the retriever return anything?
r = store.as_retriever(search_kwargs={"k": 5}).invoke("how do i return a 404")
print("retrieved   :", len(r))

if r:
    # 3 — does the retrieved metadata carry original_text, and is it non-empty?
    print("metadata keys:", list(r[0].metadata.keys()))
    print("original_text:", repr(r[0].metadata.get("original_text", "<MISSING>"))[:200])
    print("page_content :", repr(r[0].page_content)[:200])