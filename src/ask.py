import sys

from src.ingest import load_and_chunk
from src.retrievers import get_store, get_reranked
from src.generate import answer

if len(sys.argv) < 2:
    print('usage: python -m src.ask "your question"')
    raise SystemExit(1)

question = " ".join(sys.argv[1:])

docs = load_and_chunk()
retriever = get_reranked(get_store(docs), k_wide=15, top_n=8)

result = answer(question, retriever)

print(result['answer'])
print("\n Sources:")
for s in result['sources']:
    print(f"  [{s['n']}] {s['heading']}")
    print(f"       {s['url']}")