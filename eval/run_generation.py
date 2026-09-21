import json

from src.ingest import load_and_chunk
from src.retrievers import get_store, get_reranked
from src.generate import answer

docs = load_and_chunk
questions = json.load(open("eval/questions.json"))
retriever = get_reranked(get_store(docs), k_wide=15, top_n=8)

answerable = [q for q in questions if q['answerable']]
unanswerable = [q for q in questions if not q['answerable']]

print("=== should NOT refuse ===")
false_refusals = 0
for q in answerable:
     r = answer(q["q"], retriever)
     if r["refused"]:
        false_refusals += 1
     print(f"{'REFUSED' if r['refused'] else 'answered':9}  {q['q'][:45]}")

print("\n=== should refuse ===")
correct_refusals = 0
for q in unanswerable:
    r = answer(q["q"], retriever)
    if r["refused"]:
        correct_refusals += 1
    print(f"{'refused' if r['refused'] else 'ANSWERED':9}  {q['q'][:45]}")
    if not r["refused"]:
        print(f"           → {r['answer'][:120]}")

print()
print(f"correct refusal:  {correct_refusals}/{len(unanswerable)}")
print(f"false refusal:    {false_refusals}/{len(answerable)}")