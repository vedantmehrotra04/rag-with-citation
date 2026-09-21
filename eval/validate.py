import json
from src.ingest import load_and_chunk

docs = load_and_chunk()
questions = json.load(open("eval/questions.json"))

for q in questions:
    if not q['answerable']:
        continue
    phrase = q['must_contain'].lower()
    hits = [d for d in docs if phrase in d.metadata['original_text'].lower()]
    print(f"{len(hits):3} {q['q'][:50]:52} | {phrase}")