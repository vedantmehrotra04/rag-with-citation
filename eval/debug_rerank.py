import json

from src.ingest import load_and_chunk
from src.retrievers import get_store, _get_cross_encoder

docs = load_and_chunk()
store = get_store(docs)
model = _get_cross_encoder()

q = "how do i get the id from the url"
want = ("docs/tutorial/path-params.md", "path-parameters")

print(model.score([
    ("how do i get the id from the url",
     "You can declare path parameters with the same syntax used by Python format strings. "
     "The value of the path parameter item_id will be passed to your function as item_id."),
]))

# cands = store.as_retriever(search_kwargs={"k": 30}).invoke(q)
# print(f"candidates: {len(cands)}")

# scores = model.score([(q, d.page_content) for d in cands])
# scored = sorted(zip(scores, cands), key=lambda x: -x[0])

# print("\ntop 8 after reranking:")
# for s, d in scored[:8]:
#     sec = (d.metadata["doc_id"], d.metadata["anchor"])
#     mark = "  <-- CORRECT" if sec == want else ""
#     print(f"  {s:8.4f}  {sec}{mark}")

# print("\nwhere the correct one landed:")
# for rank, (s, d) in enumerate(scored, start=1):
#     if (d.metadata["doc_id"], d.metadata["anchor"]) == want:
#         print(f"  rank {rank} of {len(scored)}, score {s:.4f}")