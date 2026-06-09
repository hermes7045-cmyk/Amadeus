from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from ..models import (
    FileInput,
    ImageInput,
    LLMMessage,
    MessageContent,
    TEXT_CONTENT_BLOCK_TYPE,
    build_user_message_content,
    is_message_content_empty,
    message_content_blocks,
    message_content_to_text,
)
from ..runtime.sessions import ChatSession
from .jobs import JOB_CANCELLED_TEXT, OrchestratedTurnResult
from .roleplay import RoleplayEngine, ScheduledCronJobInfo, StreamCallback
from .roles import RoleCard


AGENT_HANDOFF_MAX_MESSAGES = 6
AGENT_HANDOFF_MAX_TOTAL_CHARS = 6000
AGENT_HANDOFF_MAX_MESSAGE_CHARS = 1800
PENDING_USER_INPUT_METADATA_KEY = "pending_user_input"


async def _discard_stream_chunk(_chunk: str) -> None:
    return None


def _build_visible_response_content(
    text: str = "",
    *,
    base_content: MessageContent = "",
    outbound_content_blocks: list[dict[str, Any]] | None = None,
) -> MessageContent:
    blocks: list[dict[str, Any]] = []
    cleaned_text = str(text or "").strip()
    if cleaned_text:
        blocks.append(
            {
                "type": TEXT_CONTENT_BLOCK_TYPE,
                "text": cleaned_text,
            }
        )
    blocks.extend(message_content_blocks(base_content))
    blocks.extend(message_content_blocks(outbound_content_blocks or []))
    if not blocks:
        return ""
    if len(blocks) == 1 and blocks[0].get("type") == TEXT_CONTENT_BLOCK_TYPE:
        return str(blocks[0].get("text", ""))
    return blocks


def _forced_agent_decision_for_pending_input():
    return SimpleNamespace(
        requires_agent=True,
        route="agent",
        reason="Continue paused task after request_user_input",
    )


def _build_agent_handoff_text(
    *,
    session: ChatSession,
) -> str | None:
    entries = _collect_handoff_entries(session.history)
    if not entries:
        return None

    lines = [
        "Visible conversation handoff from the lightweight roleplay layer.",
        "",
        "The current user request follows immediately after this handoff.",
        "Use the visible context below to resolve references such as 'that script', 'the previous result', or 'the list above'.",
        "Treat these messages as user-visible conversation context. If they mention files, memory, schedules, or tool results, verify them with tools before relying on them.",
        f"Session name: {session.name}",
        "",
        "Recent visible messages:",
    ]
    for index, entry in enumerate(entries, start=1):
        lines.append(
            f'<visible_message index="{index}" role="{entry.role}">'
        )
        lines.append(entry.content)
        lines.append("</visible_message>")
        lines.append("")
    return "\n".join(lines).strip()


def _normalize_pending_user_input(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    prompt = str(value.get("prompt", "")).strip()
    if not prompt:
        return None

    raw_choices = value.get("choices", [])
    if isinstance(raw_choices, list):
        choices = [str(choice).strip() for choice in raw_choices if str(choice).strip()]
    else:
        choices = []

    why_needed = str(value.get("why_needed", "")).strip()
    normalized = {
        "prompt": prompt,
        "choices": choices,
        "why_needed": why_needed,
    }
    return normalized


def _pending_user_input_from_metadata(
    metadata: dict[str, object] | None,
) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(PENDING_USER_INPUT_METADATA_KEY)
    if not isinstance(value, dict):
        return None
    return _normalize_pending_user_input(dict(value))


def _set_pending_user_input(
    metadata: dict[str, object],
    pending_user_input: dict[str, Any],
) -> dict[str, object]:
    next_metadata = dict(metadata or {})
    next_metadata[PENDING_USER_INPUT_METADATA_KEY] = dict(pending_user_input)
    return next_metadata


def _clear_pending_user_input(
    metadata: dict[str, object],
) -> dict[str, object]:
    next_metadata = dict(metadata or {})
    next_metadata.pop(PENDING_USER_INPUT_METADATA_KEY, None)
    return next_metadata


def _build_pending_user_input_continuation_text(
    pending_user_input: dict[str, Any] | None,
) -> str | None:
    normalized = _normalize_pending_user_input(pending_user_input)
    if normalized is None:
        return None

    lines = [
        "The previous agent run paused to request missing information.",
        "Treat the current user message as the latest reply to that request or as a new instruction that supersedes it.",
        "Continue the task with the hidden agent history from the previous run.",
        "",
        "Pending user input request:",
        normalized["prompt"],
    ]
    choices = normalized.get("choices") or []
    if choices:
        lines.append("")
        lines.append("Suggested choices:")
        lines.extend(f"- {choice}" for choice in choices)
    why_needed = str(normalized.get("why_needed", "")).strip()
    if why_needed:
        lines.append("")
        lines.append("Why this was needed:")
        lines.append(why_needed)
    return "\n".join(lines).strip()


@dataclass(slots=True)
class _HandoffEntry:
    role: str
    content: str


def _collect_handoff_entries(history: list[LLMMessage]) -> list[_HandoffEntry]:
    selected: list[_HandoffEntry] = []
    remaining_chars = AGENT_HANDOFF_MAX_TOTAL_CHARS
    for message in reversed(history):
        if message.role not in {"user", "assistant"}:
            continue
        content = message.content_text.strip()
        if not content:
            continue
        if remaining_chars <= 0:
            break
        trimmed = _trim_handoff_content(
            content,
            max_chars=min(AGENT_HANDOFF_MAX_MESSAGE_CHARS, remaining_chars),
        )
        if not trimmed:
            continue
        selected.append(_HandoffEntry(role=message.role, content=trimmed))
        remaining_chars -= len(trimmed)
        if len(selected) >= AGENT_HANDOFF_MAX_MESSAGES:
            break
    selected.reverse()
    return selected


def _trim_handoff_content(content: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    stripped = content.strip()
    if len(stripped) <= max_chars:
        return stripped
    if max_chars <= 16:
        return stripped[:max_chars]
    return stripped[: max_chars - 16].rstrip() + "\n...[truncated]"


def _extract_scheduled_cron_job(
    messages: list[LLMMessage],
) -> ScheduledCronJobInfo | None:
    cron_add_calls: dict[str, str] = {}
    for message in messages:
        if message.role != "assistant":
            continue
        for tool_call in message.tool_calls:
            if tool_call.name != "cron":
                continue
            arguments = _try_parse_json_object(tool_call.arguments)
            if not isinstance(arguments, dict):
                continue
            action = str(arguments.get("action", "")).strip().lower()
            if action != "add":
                continue
            cron_add_calls[tool_call.id] = str(arguments.get("content", "")).strip()

    if not cron_add_calls:
        return None

    for message in messages:
        if message.role != "tool":
            continue
        if message.tool_call_id not in cron_add_calls:
            continue
        payload = _try_parse_json_object(message.content)
        if not isinstance(payload, dict) or not payload.get("ok"):
            continue
        result = payload.get("result")
        if not isinstance(result, dict) or not result.get("created"):
            continue
        job = result.get("job")
        if not isinstance(job, dict):
            continue
        return ScheduledCronJobInfo(
            name=str(job.get("name", "")).strip(),
            schedule=str(job.get("schedule", "")).strip(),
            next_run_at=_optional_text(job.get("next_run_at")),
            payload_kind=str(job.get("payload_kind", "")).strip(),
            payload_content=cron_add_calls.get(message.tool_call_id, ""),
        )

    return None


def _try_parse_json_object(text: str) -> dict[str, object] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class StreamOrchestrator:
    """Orchestrates streaming output processing for conversation turns.

    Owns the streaming-specific concerns: invoking the roleplay engine's
    streaming methods, feeding chunks to the caller's callback, saving
    the completed session history, and presenting final visible results.
    """

    def __init__(
        self,
        *,
        session_store: Any,
        roleplay_engine: RoleplayEngine,
    ) -> None:
        self._session_store = session_store
        self._roleplay_engine = roleplay_engine

    async def stream_roleplay_turn(
        self,
        session: ChatSession,
        prompt: str,
        *,
        image_urls: list[ImageInput] | None = None,
        file_attachments: list[FileInput] | None = None,
        role_card: RoleCard,
        on_chunk: StreamCallback | None = None,
    ) -> OrchestratedTurnResult:
        """Handle a roleplay-only turn with streaming output.

        Streams the roleplay reply via *on_chunk*, persists the user and
        assistant messages to the session, and returns an
        ``OrchestratedTurnResult`` with ``completed=True`` and
        ``delegated=False``.
        """
        chunk_handler = on_chunk or _discard_stream_chunk

        response_text = await self._roleplay_engine.stream_chat_reply(
            session=session,
            user_input=prompt,
            image_urls=image_urls,
            file_attachments=file_attachments,
            role_card=role_card,
            on_chunk=chunk_handler,
        )

        session.history.extend(
            [
                LLMMessage(
                    role="user",
                    content=build_user_message_content(
                        prompt,
                        image_urls,
                        file_attachments,
                    ),
                ),
                LLMMessage(role="assistant", content=response_text),
            ]
        )
        await asyncio.to_thread(self._session_store.save_session, session)

        return OrchestratedTurnResult(
            session=session,
            response_text=response_text,
            delegated=False,
            completed=True,
            response_content=response_text,
            role_name=role_card.name,
            steps=1,
            compressed_summary=session.compressed_summary,
        )

    async def finalize_visible_result(
        self,
        session_name: str,
        *,
        prompt: str,
        image_urls: list[ImageInput] | None = None,
        file_attachments: list[FileInput] | None = None,
        raw_content: str,
        is_error: bool,
        scheduled_job: ScheduledCronJobInfo | None = None,
        outbound_content_blocks: list[dict[str, Any]] | None = None,
        bypass_roleplay: bool = False,
        direct_response_content: MessageContent = "",
        pending_user_input: dict[str, Any] | None = None,
        session_lock_fn: Callable[[str], Awaitable[asyncio.Lock]],
        is_session_deleted_fn: Callable[[str], Awaitable[bool]],
        resolve_session_role_fn: Callable[[ChatSession], RoleCard],
    ) -> tuple[str, MessageContent, str]:
        """Finalize the visible result after an agent execution.

        Acquires the session lock, loads/appends to the session, applies
        roleplay presentation, and persists. Returns the final text,
        content, and visible role name.
        """
        lock = await session_lock_fn(session_name)
        async with lock:
            deleted = await is_session_deleted_fn(session_name)
            if deleted:
                return "", "", ""
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            role_card = resolve_session_role_fn(session)
            metadata_changed = False
            normalized_pending_user_input = _normalize_pending_user_input(
                pending_user_input
            )
            if normalized_pending_user_input is not None:
                session.metadata = _set_pending_user_input(
                    session.metadata,
                    normalized_pending_user_input,
                )
                metadata_changed = True
            if bypass_roleplay:
                lead_in_text = ""
                if normalized_pending_user_input is not None:
                    lead_in_text = await self._roleplay_engine.present_user_input_request(
                        session=session,
                        follow_up_prompt=normalized_pending_user_input["prompt"],
                        choices=list(normalized_pending_user_input.get("choices") or []),
                        why_needed=str(
                            normalized_pending_user_input.get("why_needed", "")
                        ),
                        role_card=role_card,
                    )
                final_content = _build_visible_response_content(
                    text=lead_in_text,
                    base_content=direct_response_content,
                    outbound_content_blocks=outbound_content_blocks,
                )
            elif raw_content.strip():
                if is_error:
                    final_text = await self._roleplay_engine.present_agent_failure(
                        session=session,
                        user_input=prompt,
                        error_text=raw_content,
                        image_urls=image_urls,
                        file_attachments=file_attachments,
                        role_card=role_card,
                    )
                elif scheduled_job is not None:
                    final_text = (
                        await self._roleplay_engine.present_scheduled_setup_result(
                            session=session,
                            user_input=prompt,
                            agent_output=raw_content,
                            image_urls=image_urls,
                            file_attachments=file_attachments,
                            scheduled_job=scheduled_job,
                            role_card=role_card,
                        )
                    )
                else:
                    final_text = await self._roleplay_engine.present_agent_result(
                        session=session,
                        user_input=prompt,
                        agent_output=raw_content,
                        image_urls=image_urls,
                        file_attachments=file_attachments,
                        role_card=role_card,
                    )
            else:
                final_text = ""

            if not bypass_roleplay:
                final_content = _build_visible_response_content(
                    final_text,
                    outbound_content_blocks=outbound_content_blocks,
                )
            appended_message = False
            if not is_message_content_empty(final_content):
                session.history.append(
                    LLMMessage(role="assistant", content=final_content)
                )
                appended_message = True
            if appended_message or metadata_changed:
                await asyncio.to_thread(self._session_store.save_session, session)
            return (
                message_content_to_text(final_content).strip(),
                final_content,
                role_card.name,
            )

    async def present_scheduled_notification(
        self,
        session_name: str,
        raw_content: str,
        *,
        session_lock_fn: Callable[[str], Awaitable[asyncio.Lock]],
        is_session_deleted_fn: Callable[[str], Awaitable[bool]],
        resolve_session_role_fn: Callable[[ChatSession], RoleCard],
    ) -> str:
        """Present a scheduled notification to the session."""
        content = raw_content.strip()
        if not content:
            return ""

        lock = await session_lock_fn(session_name)
        async with lock:
            deleted = await is_session_deleted_fn(session_name)
            if deleted:
                return ""
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            role_card = resolve_session_role_fn(session)
            final_text = await self._roleplay_engine.present_scheduled_notification(
                session=session,
                reminder_text=content,
                role_card=role_card,
            )
            if final_text.strip():
                session.history.append(LLMMessage(role="assistant", content=final_text))
                await asyncio.to_thread(self._session_store.save_session, session)
            return final_text

    async def append_cancelled_message(
        self,
        session_name: str,
        *,
        session_lock_fn: Callable[[str], Awaitable[asyncio.Lock]],
        is_session_deleted_fn: Callable[[str], Awaitable[bool]],
    ) -> str:
        """Append a cancellation message to the session history."""
        final_text = JOB_CANCELLED_TEXT
        lock = await session_lock_fn(session_name)
        async with lock:
            deleted = await is_session_deleted_fn(session_name)
            if deleted:
                return ""
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            session.history.append(LLMMessage(role="assistant", content=final_text))
            await asyncio.to_thread(self._session_store.save_session, session)
        return final_text

    async def prepare_agent_handoff(
        self,
        session: ChatSession,
        prompt: str,
        *,
        image_urls: list[ImageInput] | None = None,
        file_attachments: list[FileInput] | None = None,
        role_card: RoleCard,
        delegated_ack_enabled: bool = True,
        pending_user_input: dict[str, Any] | None = None,
    ) -> AgentHandoffData:
        """Prepare the session for agent delegation and return handoff data.

        Generates the immediate ack, builds handoff/continuation text,
        persists user/assistant messages, saves the session, and streams
        the immediate ack (if any) via the chunk handler.
        """
        immediate_response = ""
        if delegated_ack_enabled and pending_user_input is None:
            immediate_response = await self._roleplay_engine.delegated_ack(
                session=session,
                user_input=prompt,
                image_urls=image_urls,
                file_attachments=file_attachments,
                role_card=role_card,
            )
        handoff_text = _build_agent_handoff_text(session=session)
        continuation_text = _build_pending_user_input_continuation_text(
            pending_user_input,
        )
        session.history.append(
            LLMMessage(
                role="user",
                content=build_user_message_content(prompt, image_urls, file_attachments),
            )
        )
        if immediate_response.strip():
            session.history.append(
                LLMMessage(role="assistant", content=immediate_response)
            )
        if pending_user_input is not None:
            session.metadata = _clear_pending_user_input(session.metadata)
        await asyncio.to_thread(self._session_store.save_session, session)
        return AgentHandoffData(
            immediate_response=immediate_response,
            handoff_text=handoff_text or "",
            continuation_text=continuation_text or "",
        )


@dataclass(slots=True)
class AgentHandoffData:
    """Data returned by :meth:`StreamOrchestrator.prepare_agent_handoff`."""
    immediate_response: str
    handoff_text: str
    continuation_text: str
