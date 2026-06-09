from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..models import (
    FileInput,
    ImageInput,
    LLMMessage,
    MessageContent,
    build_user_message_content,
    is_message_content_empty,
    message_content_to_text,
)
from ..runtime.session_runner import SessionAgentRunner
from ..runtime.sessions import ChatSession, SessionStore
from .decision import DecisionEngine
from .jobs import (
    JOB_CANCELLED_TEXT,
    CompletionCallback,
    ConversationJob,
    ConversationJobStore,
    OrchestratedTurnResult,
    job_can_retry,
)
from .roleplay import RoleplayEngine, ScheduledCronJobInfo, StreamCallback
from .route_modes import (
    RouteMode,
    route_mode_from_metadata,
    set_route_mode,
)
from .roles import RoleCard, RoleCardRegistry, role_name_from_metadata, set_role_name
from .streaming import (
    AGENT_HANDOFF_MAX_MESSAGES,
    AGENT_HANDOFF_MAX_TOTAL_CHARS,
    AGENT_HANDOFF_MAX_MESSAGE_CHARS,
    PENDING_USER_INPUT_METADATA_KEY,
    AgentHandoffData,
    StreamOrchestrator,
    _HandoffEntry,
    _build_agent_handoff_text,
    _build_pending_user_input_continuation_text,
    _build_visible_response_content,
    _clear_pending_user_input,
    _collect_handoff_entries,
    _discard_stream_chunk,
    _extract_scheduled_cron_job,
    _forced_agent_decision_for_pending_input,
    _normalize_pending_user_input,
    _optional_text,
    _pending_user_input_from_metadata,
    _set_pending_user_input,
    _trim_handoff_content,
    _try_parse_json_object,
)


BackgroundJobFactory = Callable[[], Awaitable[None]]


class ConversationCoordinator:
    def __init__(
        self,
        *,
        session_store: SessionStore,
        agent_runner: SessionAgentRunner,
        decision_engine: DecisionEngine,
        roleplay_engine: RoleplayEngine,
        role_registry: RoleCardRegistry,
        delegated_ack_enabled: bool = True,
        job_store: ConversationJobStore | None = None,
        stream_orchestrator: StreamOrchestrator | None = None,
    ) -> None:
        self._session_store = session_store
        self._agent_runner = agent_runner
        self._decision_engine = decision_engine
        self._roleplay_engine = roleplay_engine
        self._role_registry = role_registry
        self._delegated_ack_enabled = delegated_ack_enabled
        self._jobs = job_store or ConversationJobStore()
        self._stream_orchestrator = stream_orchestrator or StreamOrchestrator(
            session_store=session_store,
            roleplay_engine=roleplay_engine,
        )
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_locks_guard = asyncio.Lock()
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._job_tasks: dict[str, asyncio.Task[None]] = {}
        self._deleted_sessions: set[str] = set()
        self._deleted_sessions_guard = asyncio.Lock()

    @property
    def delegated_ack_enabled(self) -> bool:
        return self._delegated_ack_enabled

    def set_delegated_ack_enabled(self, enabled: bool) -> None:
        self._delegated_ack_enabled = bool(enabled)

    async def handle_user_turn(
        self,
        session_name: str,
        prompt: str,
        *,
        image_urls: list[ImageInput] | None = None,
        file_attachments: list[FileInput] | None = None,
        role_name: str | None = None,
        route_mode: RouteMode | None = None,
        completion_callback: CompletionCallback | None = None,
        retry_of_job_id: str | None = None,
        attempt: int = 1,
    ) -> OrchestratedTurnResult:
        return await self.handle_user_turn_stream(
            session_name,
            prompt,
            image_urls=image_urls,
            file_attachments=file_attachments,
            role_name=role_name,
            route_mode=route_mode,
            completion_callback=completion_callback,
            retry_of_job_id=retry_of_job_id,
            attempt=attempt,
        )

    async def handle_user_turn_stream(
        self,
        session_name: str,
        prompt: str,
        *,
        image_urls: list[ImageInput] | None = None,
        file_attachments: list[FileInput] | None = None,
        role_name: str | None = None,
        route_mode: RouteMode | None = None,
        completion_callback: CompletionCallback | None = None,
        on_chunk: StreamCallback | None = None,
        retry_of_job_id: str | None = None,
        attempt: int = 1,
    ) -> OrchestratedTurnResult:
        await self.restore_session(session_name)
        chunk_handler = on_chunk or _discard_stream_chunk
        lock = await self._session_lock(session_name)
        async with lock:
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            role_card = self._resolve_turn_role(session, role_name)
            resolved_route_mode = self._resolve_turn_route_mode(session, route_mode)
            pending_user_input = _pending_user_input_from_metadata(session.metadata)
            if pending_user_input is None:
                decision = await self._decision_engine.decide(
                    prompt,
                    history=list(session.history[-8:]),
                    route_mode=resolved_route_mode,
                )
            else:
                decision = _forced_agent_decision_for_pending_input()

            if not decision.requires_agent:
                return await self._stream_orchestrator.stream_roleplay_turn(
                    session,
                    prompt,
                    image_urls=image_urls,
                    file_attachments=file_attachments,
                    role_card=role_card,
                    on_chunk=chunk_handler,
                )

            handoff_data = await self._stream_orchestrator.prepare_agent_handoff(
                session,
                prompt,
                image_urls=image_urls,
                file_attachments=file_attachments,
                role_card=role_card,
                delegated_ack_enabled=self._delegated_ack_enabled,
                pending_user_input=pending_user_input,
            )
            create_trace_run_id = getattr(self._agent_runner, "create_trace_run_id", None)
            trace_run_id = (
                create_trace_run_id()
                if callable(create_trace_run_id)
                else None
            )
            job = await self._jobs.create(
                session_name=session.name,
                prompt=prompt,
                immediate_response=handoff_data.immediate_response,
                role_name=role_card.name,
                route_mode=resolved_route_mode,
                image_urls=image_urls or [],
                file_attachments=file_attachments or [],
                trace_run_id=trace_run_id,
                attempt=attempt,
                retry_of_job_id=retry_of_job_id,
            )
            self._start_background_job(
                job.job_id,
                lambda: self._run_agent_job(
                    job.job_id,
                    session_name=session.name,
                    prompt=prompt,
                    image_urls=image_urls,
                    file_attachments=file_attachments,
                    handoff_text=handoff_data.handoff_text,
                    continuation_text=handoff_data.continuation_text,
                    trace_run_id=trace_run_id,
                    completion_callback=completion_callback,
                ),
            )
            if handoff_data.immediate_response.strip():
                await chunk_handler(handoff_data.immediate_response)
            return OrchestratedTurnResult(
                session=session,
                response_text=handoff_data.immediate_response,
                delegated=True,
                completed=False,
                response_content=handoff_data.immediate_response,
                job_id=job.job_id,
                status=job.status,
                role_name=role_card.name,
                steps=0,
                compressed_summary=session.compressed_summary,
            )

    async def load_session(self, session_name: str) -> ChatSession:
        lock = await self._session_lock(session_name)
        async with lock:
            return await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )

    async def set_session_role(self, session_name: str, role_name: str) -> ChatSession:
        role_card = self._role_registry.require(role_name)
        lock = await self._session_lock(session_name)
        async with lock:
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            session.metadata = set_role_name(session.metadata, role_card.name)
            await asyncio.to_thread(self._session_store.save_session, session)
            return session

    async def current_role_name(self, session_name: str) -> str:
        lock = await self._session_lock(session_name)
        async with lock:
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            role_name = role_name_from_metadata(session.metadata)
            try:
                role_card = self._role_registry.require(role_name)
            except ValueError:
                role_card = self._role_registry.require(None)
                session.metadata = set_role_name(session.metadata, role_card.name)
                await asyncio.to_thread(self._session_store.save_session, session)
            return role_card.name

    async def set_session_route_mode(
        self,
        session_name: str,
        route_mode: RouteMode,
    ) -> ChatSession:
        lock = await self._session_lock(session_name)
        async with lock:
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            session.metadata = set_route_mode(session.metadata, route_mode)
            await asyncio.to_thread(self._session_store.save_session, session)
            return session

    async def current_route_mode(self, session_name: str) -> RouteMode:
        lock = await self._session_lock(session_name)
        async with lock:
            session = await asyncio.to_thread(
                self._session_store.load_or_create_session,
                session_name,
            )
            route_mode = route_mode_from_metadata(session.metadata)
            session.metadata = set_route_mode(session.metadata, route_mode)
            await asyncio.to_thread(self._session_store.save_session, session)
            return route_mode

    def available_roles(self) -> list[str]:
        return self._role_registry.names()

    async def get_job(self, job_id: str) -> ConversationJob | None:
        return await self._jobs.get(job_id)

    async def list_jobs(
        self,
        *,
        session_name: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[ConversationJob]:
        return await self._jobs.list_jobs(
            session_name=session_name,
            status=status,
            limit=limit,
        )

    async def retry_job(
        self,
        job_id: str,
        *,
        completion_callback: CompletionCallback | None = None,
    ) -> OrchestratedTurnResult:
        job = await self._jobs.get(job_id)
        if job is None:
            raise ValueError(f"任务不存在：{job_id}")
        if not job_can_retry(job):
            raise ValueError("只有失败或已取消的任务可以重试")

        route_mode = (
            job.route_mode
            if job.route_mode in {"auto", "chat_only", "force_agent"}
            else None
        )
        return await self.handle_user_turn(
            job.session_name,
            job.prompt,
            image_urls=job.image_urls,
            file_attachments=job.file_attachments,
            role_name=job.role_name,
            route_mode=route_mode,
            completion_callback=completion_callback,
            retry_of_job_id=job.job_id,
            attempt=job.attempt + 1,
        )

    async def get_job_trace(
        self,
        job_id: str,
    ) -> tuple[ConversationJob | None, list[dict[str, Any]]]:
        job = await self._jobs.get(job_id)
        if job is None:
            return None, []
        if not job.trace_run_id:
            return job, []
        events = await self._agent_runner.load_trace_events(
            job.session_name,
            job.trace_run_id,
        )
        return job, events

    async def cancel_job(self, job_id: str) -> ConversationJob | None:
        job = await self._jobs.get(job_id)
        if job is None:
            return None
        if job.status != "running":
            return job

        task = self._job_tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        job = await self._jobs.get(job_id)
        if job is None:
            return None
        if job.status != "running":
            return job

        final_text = await self._append_cancelled_message(job.session_name)
        return await self._jobs.set_cancelled(
            job_id,
            final_response=final_text,
            final_response_content=final_text,
        )

    async def cancel_jobs_for_session(self, session_name: str) -> list[ConversationJob]:
        running_jobs = await self._jobs.list_for_session(
            session_name,
            status="running",
        )
        if not running_jobs:
            return []

        tasks: list[asyncio.Task[None]] = []
        for job in running_jobs:
            task = self._job_tasks.get(job.job_id)
            if task is None or task.done():
                continue
            task.cancel()
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        cancelled_jobs: list[ConversationJob] = []
        for job in running_jobs:
            current_job = await self._jobs.get(job.job_id)
            if current_job is None or current_job.status != "running":
                if current_job is not None:
                    cancelled_jobs.append(current_job)
                continue
            updated_job = await self._jobs.set_cancelled(
                job.job_id,
                final_response="",
                final_response_content="",
            )
            if updated_job is not None:
                cancelled_jobs.append(updated_job)
        return cancelled_jobs

    async def mark_session_deleted(self, session_name: str) -> None:
        async with self._deleted_sessions_guard:
            self._deleted_sessions.add(session_name)
        mark_deleted = getattr(self._agent_runner, "mark_session_deleted", None)
        if mark_deleted is not None:
            await mark_deleted(session_name)

    async def restore_session(self, session_name: str) -> None:
        async with self._deleted_sessions_guard:
            self._deleted_sessions.discard(session_name)
        restore = getattr(self._agent_runner, "restore_session", None)
        if restore is not None:
            await restore(session_name)

    async def job_counts(self) -> dict[str, int]:
        return await self._jobs.counts()

    async def close(self) -> None:
        job_ids = list(self._job_tasks)
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for job_id in job_ids:
            await self._mark_job_cancelled(job_id)
        self._background_tasks.clear()
        self._job_tasks.clear()

    async def _run_agent_job(
        self,
        job_id: str,
        *,
        session_name: str,
        prompt: str,
        image_urls: list[ImageInput] | None,
        file_attachments: list[FileInput] | None,
        handoff_text: str | None,
        continuation_text: str | None,
        trace_run_id: str | None,
        completion_callback: CompletionCallback | None,
    ) -> None:
        visible_role_name = ""
        try:
            run_prompt_kwargs: dict[str, Any] = {
                "scheduled_context": False,
                "transient_system_messages": (
                    [
                        message
                        for message in [continuation_text, handoff_text]
                        if message and message.strip()
                    ]
                    or None
                ),
                "image_urls": image_urls,
                "file_attachments": file_attachments,
                "trace_run_id": trace_run_id,
            }

            execution = await self._agent_runner.run_prompt(
                session_name,
                prompt,
                **run_prompt_kwargs,
            )
            raw_content = message_content_to_text(
                execution.agent_result.response.message.content
            ).strip()
            outbound_content_blocks = list(
                execution.agent_result.outbound_content_blocks or []
            )
            awaiting_user_input = execution.agent_result.status == "waiting_for_input"
            scheduled_job = _extract_scheduled_cron_job(
                execution.agent_result.new_messages,
            )
            final_text, final_content, visible_role_name = await self._finalize_visible_result(
                session_name,
                prompt=prompt,
                image_urls=image_urls,
                file_attachments=file_attachments,
                raw_content=raw_content,
                is_error=False,
                scheduled_job=scheduled_job,
                outbound_content_blocks=outbound_content_blocks,
                bypass_roleplay=awaiting_user_input,
                direct_response_content=(
                    execution.agent_result.response.message.content
                    if awaiting_user_input
                    else ""
                ),
                pending_user_input=execution.agent_result.pending_user_input,
            )
            if awaiting_user_input:
                job = await self._jobs.set_waiting_for_input(
                    job_id,
                    final_response=final_text,
                    final_response_content=final_content,
                    steps=execution.agent_result.steps,
                    pending_user_input=execution.agent_result.pending_user_input,
                )
            else:
                job = await self._jobs.set_completed(
                    job_id,
                    final_response=final_text,
                    final_response_content=final_content,
                    steps=execution.agent_result.steps,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error_text = str(exc)
            final_text, final_content, visible_role_name = await self._finalize_visible_result(
                session_name,
                prompt=prompt,
                image_urls=image_urls,
                file_attachments=file_attachments,
                raw_content=error_text,
                is_error=True,
            )
            job = await self._jobs.set_failed(
                job_id,
                final_response=final_text,
                final_response_content=final_content,
                error=error_text,
            )

        if job is None:
            return
        if not job.final_response.strip():
            return
        if completion_callback is None:
            return

        await completion_callback(
            ConversationJob(
                job_id=job.job_id,
                session_name=job.session_name,
                prompt=job.prompt,
                immediate_response=job.immediate_response,
                role_name=visible_role_name or job.role_name,
                status=job.status,
                created_at=job.created_at,
                updated_at=job.updated_at,
                started_at=job.started_at,
                finished_at=job.finished_at,
                trace_run_id=job.trace_run_id,
                route_mode=job.route_mode,
                attempt=job.attempt,
                retry_of_job_id=job.retry_of_job_id,
                image_urls=job.image_urls,
                file_attachments=job.file_attachments,
                final_response=job.final_response,
                final_response_content=job.final_response_content,
                error=job.error,
                steps=job.steps,
                pending_user_input=job.pending_user_input,
            )
        )

    async def _finalize_visible_result(
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
    ) -> tuple[str, MessageContent, str]:
        return await self._stream_orchestrator.finalize_visible_result(
            session_name,
            prompt=prompt,
            image_urls=image_urls,
            file_attachments=file_attachments,
            raw_content=raw_content,
            is_error=is_error,
            scheduled_job=scheduled_job,
            outbound_content_blocks=outbound_content_blocks,
            bypass_roleplay=bypass_roleplay,
            direct_response_content=direct_response_content,
            pending_user_input=pending_user_input,
            session_lock_fn=self._session_lock,
            is_session_deleted_fn=self._is_session_deleted,
            resolve_session_role_fn=self._resolve_session_role,
        )

    async def present_scheduled_notification(
        self,
        session_name: str,
        raw_content: str,
    ) -> str:
        return await self._stream_orchestrator.present_scheduled_notification(
            session_name,
            raw_content,
            session_lock_fn=self._session_lock,
            is_session_deleted_fn=self._is_session_deleted,
            resolve_session_role_fn=self._resolve_session_role,
        )

    async def _append_cancelled_message(self, session_name: str) -> str:
        return await self._stream_orchestrator.append_cancelled_message(
            session_name,
            session_lock_fn=self._session_lock,
            is_session_deleted_fn=self._is_session_deleted,
        )

    def _resolve_turn_role(
        self,
        session: ChatSession,
        role_name: str | None,
    ) -> RoleCard:
        if role_name is not None:
            role_card = self._role_registry.require(role_name)
            session.metadata = set_role_name(session.metadata, role_card.name)
            return role_card
        return self._resolve_session_role(session)

    def _resolve_session_role(self, session: ChatSession) -> RoleCard:
        role_name = role_name_from_metadata(session.metadata)
        try:
            role_card = self._role_registry.require(role_name)
        except ValueError:
            role_card = self._role_registry.require(None)
        session.metadata = set_role_name(session.metadata, role_card.name)
        return role_card

    def _resolve_turn_route_mode(
        self,
        session: ChatSession,
        route_mode: RouteMode | None,
    ) -> RouteMode:
        if route_mode is not None:
            return route_mode
        return route_mode_from_metadata(session.metadata)

    async def _mark_job_cancelled(self, job_id: str) -> None:
        job = await self._jobs.get(job_id)
        if job is None or job.status != "running":
            return
        await self._jobs.set_cancelled(
            job_id,
            final_response=JOB_CANCELLED_TEXT,
            final_response_content=JOB_CANCELLED_TEXT,
        )

    async def _session_lock(self, session_name: str) -> asyncio.Lock:
        async with self._session_locks_guard:
            lock = self._session_locks.get(session_name)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[session_name] = lock
            return lock

    async def _is_session_deleted(self, session_name: str) -> bool:
        async with self._deleted_sessions_guard:
            return session_name in self._deleted_sessions

    def _start_background_job(
        self,
        job_id: str,
        coroutine_factory: BackgroundJobFactory,
    ) -> None:
        task = asyncio.create_task(
            self._run_after_yield(coroutine_factory),
            name=f"amadeus_conversation_job_{job_id}",
        )
        self._background_tasks.add(task)
        self._job_tasks[job_id] = task

        def cleanup(done_task: asyncio.Task[None]) -> None:
            self._background_tasks.discard(done_task)
            if self._job_tasks.get(job_id) is done_task:
                self._job_tasks.pop(job_id, None)

        task.add_done_callback(cleanup)

    async def _run_after_yield(
        self,
        coroutine_factory: BackgroundJobFactory,
    ) -> None:
        await asyncio.sleep(0)
        await coroutine_factory()




