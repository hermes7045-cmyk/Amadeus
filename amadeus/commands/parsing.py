from __future__ import annotations


def split_command_parts(text: str) -> tuple[str, str]:
    cleaned = text.strip()
    if not cleaned:
        return "", ""

    parts = cleaned.split(maxsplit=1)
    raw_token = parts[0].strip().lower()
    remainder = parts[1].strip() if len(parts) >= 2 else ""
    command_token = raw_token.split("@", 1)[0]
    return command_token, remainder


def split_action_argument(
    text: str,
    *,
    lowercase_argument: bool = False,
) -> tuple[str, str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "", ""

    parts = cleaned.split(maxsplit=1)
    action = parts[0].strip().lower()
    argument = parts[1].strip() if len(parts) >= 2 else ""
    if lowercase_argument:
        argument = argument.lower()
    return action, argument
