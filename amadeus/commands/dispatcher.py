from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


ContextT = TypeVar("ContextT")


CommandParser = Callable[[str], object | None]
CommandExecutor = Callable[[ContextT, object], Awaitable["CommandResult"]]


@dataclass(slots=True)
class CommandResult:
    text: str

    @classmethod
    def from_lines(cls, lines: Sequence[str]) -> "CommandResult":
        return cls(text="\n".join(lines))


@dataclass(slots=True)
class TextCommandHandler(Generic[ContextT]):
    parse: CommandParser
    execute: CommandExecutor


BoundTextCommand = TextCommandHandler


async def dispatch_text_command(
    text: str,
    context: ContextT,
    handlers: Sequence[TextCommandHandler[ContextT]],
) -> CommandResult | None:
    for handler in handlers:
        command = handler.parse(text)
        if command is None:
            continue
        return await handler.execute(context, command)
    return None
