from langchain_core.documents import Document
import statistics
import time

def first_hit_rank(q, retrieved) -> int | None:
    want = {tuple(s) for s in q['relevant_sections']}
    for rank, d in enumerate(retrieved, start=1):
        if(d.metadata['doc_id'], d.metadata['anchor']) in want:
            return rank
    return None

def evaluate(retriever, questions: list[dict], k: int = 5, verbose: bool = True, delay = 0.0) -> dict:
    ranks: list[int | None] = []
    latencies: list[float] = []

    for q in questions: 
        if not q['answerable']:
            continue
        
        t0 = time.perf_counter()
        retrieved = retriever.invoke(q['q'])[:k]
        latencies.append(time.perf_counter() - t0)

        rank = first_hit_rank(q, retrieved)
        ranks.append(rank)
        if verbose:
            mark = "✓" if rank else "✗"
            pos = f"rank {rank}" if rank else "----"
            print(f"{mark}  {pos:7}  {q['q'][:50]}")
        
        if delay: 
            time.sleep(delay)

    hits = [r for r in ranks if r is not None]

    return {
        "n": len(ranks),
        "hit_rate": round(len(hits)/ len(ranks), 3),
        "mrr": round(sum(1 / r for r in hits) / len(ranks), 3),
        "latency_p50": round(statistics.median(latencies), 3)
    }


