"""Preserve conversational conditioning when a dataset contains chat history."""


def user_messages(user: str | list[dict]) -> list[dict]:
    if isinstance(user, str):
        return [{"role": "user", "content": user}]
    if not isinstance(user, list) or not user or user[-1].get("role") != "user":
        raise ValueError("Expected user text or a conversation ending in a user turn")
    for message in user:
        if message.get("role") not in ("user", "assistant") or not isinstance(message.get("content"), str):
            raise ValueError("Expected user/assistant messages without a system prompt")
    return [dict(message) for message in user]
