"""Preserve conversational conditioning when a dataset contains chat history."""


def user_messages(user: str | list[dict]) -> list[dict]:
    if isinstance(user, str):
        return [{"role": "user", "content": user}]
    if not isinstance(user, list) or not user or len(user) % 2 != 1:
        raise ValueError("Expected user text or a conversation ending in a user turn")
    for index, message in enumerate(user):
        role = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != role or not isinstance(message.get("content"), str):
            raise ValueError("Expected alternating user/assistant messages without a system prompt")
    return [dict(message) for message in user]
