import uuid

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import GRAPH, MAX_TURNS
from agent.utils import message_text

REFUSALS = ("i can't", "i cannot", "i'm not able", "i don't have access", "idon't know")



def _entries(update: dict) -> list[dict]:
    """One node's update -> trace entries."""
    entries = []
    for m in update.get("messages", []) or []:
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                entries.append({"type": "tool_call", "name": tc["name"],
"args": tc["args"]})
            continue

        if type(m).__name__ == "ToolMessage":
            entries.append({"type": "tool_result", "name": m.name, "content":
str(m.content)})
            continue

        text = message_text(m)
        if text:
            entries.append({"type": "answer", "text": text})

    return entries


def _last_answer(messages: list) -> str:
    """The most recent thing the assistant actually said."""
    for m in reversed(messages):
        if type(m).__name__ == "AIMessage":
            text = message_text(m)
            if text:
                return text
    return ""


def _outcome(answer: str, paused: bool) -> str:
    if paused:
        return "gated"
    text = answer.strip().lower()
    if any(p in text for p in REFUSALS):
        return "refused"
    if text.endswith("?"):
        return "clarified"
    return "answered"


def _run(config: dict, payload, question: str) -> dict:
    """Stream the graph, collect the trace, read the final state."""
    steps = []
    for chunk in GRAPH.stream(payload, config, stream_mode="updates"):
        for update in chunk.values():
            if isinstance(update, dict):
                steps += _entries(update)

    snapshot = GRAPH.get_state(config)
    paused = bool(snapshot.next)

    if paused:
        steps.append({"type": "interrupt", "payload": str(snapshot.next)})

    answer = _last_answer(snapshot.values.get("messages", []))
    hit_ceiling = snapshot.values.get("step_count", 0) >= MAX_TURNS

    return {
        "input": question,
        "thread_id": config["configurable"]["thread_id"],
        "steps": steps,
        "final_answer": answer,
        "terminated": "interrupt" if paused else ("ceiling" if hit_ceiling else "answer"),
        "outcome": _outcome(answer, paused),
    }


def run_agent(question: str, thread_id: str | None = None) -> dict:
    config = {"configurable": {"thread_id": thread_id or f"run-{uuid.uuid4().hex[:8]}"}}
    payload = {"messages": [HumanMessage(question)], "step_count": 0, "pending_credit": None}
    return _run(config, payload, question)


def resume_agent(thread_id: str, decision: str) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    return _run(config, Command(resume=decision), "")