from __future__ import annotations

from dataclasses import dataclass

from ..runtime.session_service import SessionService
from ..runtime.sessions import ChatSession, SessionInfo
from .parsing import split_action_argument, split_command_parts


SESSION_COMMANDS = {"/session", "session"}


@dataclass(slots=True)
class SavedSessionCommand:
    action: str
    argument: str = ""


@dataclass(slots=True)
class SavedSessionCommandResult:
    session: ChatSession
    lines: list[str]


def parse_saved_session_command(text: str) -> SavedSessionCommand | None:
    command_token, remainder = split_command_parts(text)
    if command_token not in SESSION_COMMANDS:
        return None

    if not remainder:
        return SavedSessionCommand(action="help")

    action, argument = split_action_argument(remainder)

    if action in {"help", "list", "current", "new", "switch", "rename", "delete"}:
        return SavedSessionCommand(action=action, argument=argument)
    return SavedSessionCommand(action="help")


def is_saved_session_command(text: str) -> bool:
    return parse_saved_session_command(text) is not None


async def execute_saved_session_command(
    *,
    session_service: SessionService,
    current_session: ChatSession,
    command: SavedSessionCommand,
) -> SavedSessionCommandResult:
    if command.action == "help":
        return SavedSessionCommandResult(
            session=current_session,
            lines=format_saved_session_help_lines(),
        )

    if command.action == "list":
        sessions = await session_service.list_sessions()
        return SavedSessionCommandResult(
            session=current_session,
            lines=format_saved_session_list_lines(
                sessions,
                current_session_name=current_session.name,
            ),
        )

    if command.action == "current":
        return SavedSessionCommandResult(
            session=current_session,
            lines=[
                (
                    f"Current session: {current_session.name} "
                    f"({len(current_session.history)} messages)"
                )
            ],
        )

    if command.action == "new":
        next_session = await session_service.create_session(command.argument or None)
        return SavedSessionCommandResult(
            session=next_session,
            lines=[f"Switched to new session: {next_session.name}"],
        )

    if command.action == "switch":
        if not command.argument:
            raise ValueError("Usage: /session switch <name>")

        next_session = await session_service.switch_session(command.argument)
        return SavedSessionCommandResult(
            session=next_session,
            lines=[
                (
                    f"Switched to session: {next_session.name} "
                    f"({len(next_session.history)} messages)"
                )
            ],
        )

    if command.action == "rename":
        if not command.argument:
            raise ValueError("Usage: /session rename <name>")

        renamed_session = await session_service.rename_session(
            current_session.name,
            command.argument,
        )
        return SavedSessionCommandResult(
            session=renamed_session,
            lines=[f"Renamed current session to: {renamed_session.name}"],
        )

    if command.action == "delete":
        sessions = await session_service.list_sessions()
        deleted = await session_service.delete_session(current_session.name)
        if not deleted:
            raise ValueError(f"Session not found: {current_session.name}")

        next_session = await session_service.load_current_session()
        if len(sessions) <= 1:
            return SavedSessionCommandResult(
                session=next_session,
                lines=[
                    "Deleted the last session and created a fresh one: "
                    f"{next_session.name}"
                ],
            )
        return SavedSessionCommandResult(
            session=next_session,
            lines=[f"Deleted current session. Now using: {next_session.name}"],
        )

    return SavedSessionCommandResult(
        session=current_session,
        lines=format_saved_session_help_lines(),
    )


def format_saved_session_help_lines() -> list[str]:
    return [
        "Session commands:",
        "- /session help",
        "- /session list",
        "- /session current",
        "- /session new [name]",
        "- /session switch <name>",
        "- /session rename <name>",
        "- /session delete",
        "- /route help",
    ]


def format_saved_session_list_lines(
    sessions: list[SessionInfo],
    *,
    current_session_name: str,
) -> list[str]:
    if not sessions:
        return ["No saved sessions."]

    lines = ["Saved sessions:"]
    for session in sessions:
        marker = "*" if session.name == current_session_name else " "
        lines.append(
            f"{marker} {session.name} | "
            f"{session.message_count} messages | "
            f"{session.updated_at}"
        )
    return lines
