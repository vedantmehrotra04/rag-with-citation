def message_text(message) -> str:
    """Flatten a message's content to a string.

    Newer models return content as a list of blocks — [{"type": "text",
    "text": "..."}] — rather than a plain string. Everything downstream
    wants a string.
    """
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)