import json
from src.ingest import load_and_chunk
from src.retrievers import get_store

docs = load_and_chunk()
questions = json.load(open("eval/questions.json"))
retriever = get_store(docs).as_retriever(search_kwargs={"k": 5})

for q in questions:
    if not q["answerable"]:
        continue
    want = {tuple(s) for s in q["relevant_sections"]}
    r = retriever.invoke(q["q"])[:5]

    print(q["q"][:50])
    print("  want:", want)
    for i, d in enumerate(r, 1):
        sec = (d.metadata["doc_id"], d.metadata["anchor"])
        print(f"  {i}. {'MATCH' if sec in want else '     '} {sec}")
    print()