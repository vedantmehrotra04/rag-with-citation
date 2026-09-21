import json
from src.ingest import load_and_chunk
from src.retrievers import get_store

docs = load_and_chunk()
questions = json.load(open("eval/questions.json"))
retriever = get_store(docs).as_retriever(search_kwargs={"k": 5})

for q in questions:
    if not q["answerable"]:
        continue

    r = retriever.invoke(q["q"])[:5]

    key = q.get("must_contain")
    if key:
        hit = any(key.lower() in d.metadata["original_text"].lower() for d in r)
    else:
        key = q.get("must_be_from")
        hit = any(d.metadata["doc_id"] == key for d in r)

    print(f"{'✓' if hit else '✗'}  {q['q'][:45]:47} | {key}")
    for d in r:
        print(f"      {d.metadata['url']}")
    print()