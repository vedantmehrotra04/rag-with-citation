from src.ingest import load_and_chunk
from src.retrievers import get_store

docs = load_and_chunk()
store = get_store(docs)
print("indexed:", store._collection.count())