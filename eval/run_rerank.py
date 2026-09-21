import json
from src.ingest import load_and_chunk
from src.retrievers import get_store, get_reranked
from src.evaluate import evaluate

docs = load_and_chunk()
questions = json.load(open('eval/questions.json'))
store = get_store(docs)

print("=== vector k=5 ===")
v = evaluate(store.as_retriever(search_kwargs={"k": 5}), questions, k=5)

print("\n=== ceiling: vector k=30 ===")
c = evaluate(store.as_retriever(search_kwargs={"k": 30}), questions, k=30)

print("\n=== reranked 30 -> 5 ===")
r = evaluate(get_reranked(store), questions, k=8, delay=7.0)

print()
print(f"{'metric':12} {'k=5':>8} {'k=30':>8} {'rerank':>8}")
for m in ("hit_rate", "mrr", "latency_p50"):
    print(f"{m:12} {v[m]:8.3f} {c[m]:8.3f} {r[m]:8.3f}")