import json
from src.ingest import load_and_chunk
from src.retrievers import get_store
from src.evaluate import evaluate

docs = load_and_chunk()
questions = json.load(open("eval/questions.json"))

store = get_store(docs)
retriever = store.as_retriever(search_kwargs={"k": 5})

print(evaluate(retriever, questions, k=5))