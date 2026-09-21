import json
from src.ingest import load_and_chunk
from src.retrievers import get_store, get_hybrid
from src.evaluate import evaluate
from langchain_community.retrievers import BM25Retriever

docs = load_and_chunk()
questions = json.load(open("eval/questions.json"))
store = get_store(docs)

vector = store.as_retriever(search_kwargs={"k": 5})
hybrid = get_hybrid(docs, store, k_each=10)
bm25_only = BM25Retriever.from_documents(docs)
bm25_only.k = 5

print("==== vector ====")
v = evaluate(vector, questions, k=5, verbose=True)

print("\n==== bm25 only ====")
b = evaluate(bm25_only, questions, k=5, verbose=True)

print("\n==== hybrid ====")
h = evaluate(hybrid, questions, k=5, verbose=True)

print()
print(f"{'metric':12} {'vector': >8} {'bm25': >8} {'hybrid': >8}")

for m in ("hit_rate", "mrr", "latency_p50"):
    print(f"{m:12} {v[m]: 8.3f} {b[m]: 8.3f} {h[m]:8.3f}")

