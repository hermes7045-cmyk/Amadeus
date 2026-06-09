from __future__ import annotations

import asyncio
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

# conftest.py と同じ sys.path 設定
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_project_root_text = str(_PROJECT_ROOT)
if _project_root_text not in sys.path:
    sys.path.insert(0, _project_root_text)

from amadeus import LLMMessage, LLMResponse, ToolCall, build_default_system_prompt
from amadeus.scheduling.cron import (
    CronJob,
    CronJobState,
    CronPayload,
    CronSchedule,
    CronService,
    CronStore,
    compute_next_run,
    describe_schedule,
    normalize_schedule,
    summarize_job,
)
from amadeus.scheduling.heartbeat import (
    DEFAULT_HEARTBEAT_TEMPLATE,
    HeartbeatService,
    has_meaningful_heartbeat_content,
    read_or_create_heartbeat_file,
    write_heartbeat_file,
)
from amadeus.cli.chat import _build_schedule_notifier
from amadeus.providers.base import LLMProvider
from amadeus.tools.cron import CronTool


class CronParserTests(unittest.TestCase):
    def test_compute_next_run_for_every_schedule(self) -> None:
        now = datetime.now().astimezone()
        next_run = compute_next_run(
            CronSchedule(kind="every", every_seconds=30),
            now=now,
        )

        self.assertIsNotNone(next_run)
        assert next_run is not None
        self.assertGreaterEqual((next_run - now).total_seconds(), 30)

    def test_compute_next_run_for_five_field_cron(self) -> None:
        now = datetime(2026, 3, 8, 8, 15).astimezone()
        next_run = compute_next_run(
            CronSchedule(kind="cron", expr="0 9 * * *"),
            now=now,
        )

        self.assertEqual(9, next_run.hour)
        self.assertEqual(0, next_run.minute)


class CronServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_add_job_persists_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "cron" / "jobs.json"
            service = CronService(store_path)

            job = await service.add_job(
                name="demo",
                schedule=CronSchedule(kind="every", every_seconds=60),
                payload=CronPayload(content="hello", session_name="demo"),
            )

            self.assertTrue(store_path.exists())
            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual(job.id, saved["jobs"][0]["id"])
            self.assertEqual("demo", saved["jobs"][0]["name"])

    async def test_running_service_executes_due_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "cron" / "jobs.json"
            called: list[str] = []

            async def on_job(job) -> str:
                called.append(job.id)
                return "done"

            service = CronService(
                store_path,
                on_job=on_job,
                poll_interval_seconds=0.05,
            )
            await service.add_job(
                name="due-now",
                schedule=CronSchedule(
                    kind="at",
                    at=(
                        datetime.now().astimezone() + timedelta(seconds=1)
                    ).isoformat(timespec="seconds"),
                ),
                payload=CronPayload(content="hello", session_name="demo"),
                delete_after_run=True,
            )
            await service.start()
            try:
                await asyncio.sleep(1.5)
            finally:
                await service.stop()

            self.assertEqual(1, len(called))

    async def test_start_prunes_expired_delete_after_run_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "cron" / "jobs.json"
            past_time = (
                datetime.now().astimezone() - timedelta(minutes=5)
            ).isoformat(timespec="seconds")
            store = CronStore(
                jobs=[
                    CronJob(
                        id="expired-delete",
                        name="expired reminder",
                        enabled=True,
                        schedule=CronSchedule(kind="at", at=past_time),
                        payload=CronPayload(content="hello", session_name="demo"),
                        state=CronJobState(next_run_at=None),
                        created_at=past_time,
                        updated_at=past_time,
                        delete_after_run=True,
                    )
                ]
            )
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text(
                json.dumps(store.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            service = CronService(store_path)
            await service.start()
            try:
                jobs = await service.list_jobs(include_disabled=True)
                status = await service.status()
            finally:
                await service.stop()

            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual([], jobs)
            self.assertEqual([], saved["jobs"])
            self.assertEqual(0, status["jobs"])

    async def test_start_disables_expired_one_time_jobs_without_delete_after_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "cron" / "jobs.json"
            past_time = (
                datetime.now().astimezone() - timedelta(minutes=5)
            ).isoformat(timespec="seconds")
            store = CronStore(
                jobs=[
                    CronJob(
                        id="expired-keep",
                        name="missed reminder",
                        enabled=True,
                        schedule=CronSchedule(kind="at", at=past_time),
                        payload=CronPayload(content="hello", session_name="demo"),
                        state=CronJobState(next_run_at=None),
                        created_at=past_time,
                        updated_at=past_time,
                        delete_after_run=False,
                    )
                ]
            )
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text(
                json.dumps(store.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            service = CronService(store_path)
            await service.start()
            try:
                enabled_jobs = await service.list_jobs()
                all_jobs = await service.list_jobs(include_disabled=True)
            finally:
                await service.stop()

            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual([], enabled_jobs)
            self.assertEqual(1, len(all_jobs))
            self.assertFalse(all_jobs[0].enabled)
            self.assertIsNone(all_jobs[0].state.next_run_at)
            self.assertFalse(saved["jobs"][0]["enabled"])
            self.assertIsNone(saved["jobs"][0]["state"]["next_run_at"])


class CronToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_cron_tool_adds_and_lists_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            tool = CronTool(service, session_name="demo")

            created = await tool.run(
                {
                    "action": "add",
                    "content": "Check todos",
                    "every_seconds": 60,
                }
            )
            listed = await tool.run({"action": "list"})

            self.assertTrue(created["created"])
            self.assertEqual("demo", created["job"]["session_name"])
            self.assertEqual(1, len(listed["jobs"]))
            self.assertEqual("agent", listed["jobs"][0]["payload_kind"])

    async def test_cron_tool_uses_delay_seconds_for_one_time_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            tool = CronTool(service, session_name="demo")

            created = await tool.run(
                {
                    "action": "add",
                    "content": "20 秒后提醒我喝水",
                    "delay_seconds": 20,
                    "task_type": "text",
                }
            )

            job = created["job"]
            saved = json.loads(service.store_path.read_text(encoding="utf-8"))
            self.assertTrue(created["created"])
            self.assertEqual("text", job["payload_kind"])
            self.assertIsNotNone(job["next_run_at"])
            self.assertEqual("at", saved["jobs"][0]["schedule"]["kind"])
            delay = int(
                (
                    datetime.fromisoformat(saved["jobs"][0]["schedule"]["at"])
                    - datetime.fromisoformat(saved["jobs"][0]["created_at"])
                ).total_seconds()
            )
            self.assertGreaterEqual(delay, 19)
            self.assertLessEqual(delay, 20)
            self.assertTrue(saved["jobs"][0]["delete_after_run"])

    async def test_cron_tool_blocks_mutation_in_scheduled_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            tool = CronTool(
                service,
                session_name="demo",
                allow_mutations=False,
            )

            with self.assertRaisesRegex(ValueError, "disabled while a scheduled task"):
                await tool.run(
                    {
                        "action": "add",
                        "content": "Check todos",
                        "every_seconds": 60,
                    }
                )


class FakeHeartbeatProvider(LLMProvider):
    def __init__(self, response: LLMResponse) -> None:
        self.response = response
        self.calls = 0

    async def generate(
        self,
        messages,
        *,
        tools=None,
        tool_choice=None,
        temperature=None,
        max_tokens=None,
    ) -> LLMResponse:
        del messages, tools, tool_choice, temperature, max_tokens
        self.calls += 1
        return self.response


class HeartbeatServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_trigger_now_skips_template_only_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            heartbeat_file = Path(temp_dir) / "HEARTBEAT.md"
            heartbeat_file.write_text(
                "# HEARTBEAT.md\n\n<!-- no tasks -->\n",
                encoding="utf-8",
            )
            provider = FakeHeartbeatProvider(
                LLMResponse(
                    message=LLMMessage(role="assistant", content=""),
                    model="fake",
                )
            )
            service = HeartbeatService(
                heartbeat_file=heartbeat_file,
                provider=provider,
            )

            result = await service.trigger_now()

            self.assertIsNone(result)
            self.assertEqual(0, provider.calls)

    async def test_trigger_now_runs_tasks_when_provider_requests_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            heartbeat_file = Path(temp_dir) / "HEARTBEAT.md"
            heartbeat_file.write_text("- [ ] Check todos\n", encoding="utf-8")
            provider = FakeHeartbeatProvider(
                LLMResponse(
                    message=LLMMessage(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="heartbeat_decision",
                                arguments='{"action":"run","tasks":"review todos"}',
                            )
                        ],
                    ),
                    model="fake",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="heartbeat_decision",
                            arguments='{"action":"run","tasks":"review todos"}',
                        )
                    ],
                )
            )
            called_with: list[str] = []
            notifications: list[str] = []

            async def on_execute(tasks: str) -> str:
                called_with.append(tasks)
                return "done"

            async def on_notify(message: str) -> None:
                notifications.append(message)

            service = HeartbeatService(
                heartbeat_file=heartbeat_file,
                provider=provider,
                on_execute=on_execute,
                on_notify=on_notify,
            )

            result = await service.trigger_now()

            self.assertEqual("done", result)
            self.assertEqual(["review todos"], called_with)
            self.assertEqual(["done"], notifications)


class SystemPromptSchedulingTests(unittest.TestCase):
    def test_default_system_prompt_can_include_scheduling_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            prompt = build_default_system_prompt(
                workspace,
                enable_scheduling=True,
                cron_store_path=workspace / ".amadeus" / "cron" / "jobs.json",
                heartbeat_file_path=workspace / "HEARTBEAT.md",
                heartbeat_interval_seconds=900,
            )

            self.assertIn("## Scheduling", prompt)
            self.assertIn("HEARTBEAT.md", prompt)
            self.assertIn("cron", prompt)

    def test_default_system_prompt_uses_hidden_heartbeat_file_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            prompt = build_default_system_prompt(
                workspace,
                enable_scheduling=True,
            )

            self.assertIn(
                str((workspace / ".amadeus" / "HEARTBEAT.md").resolve()),
                prompt,
            )


class ScheduleNotifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_notifier_prints_single_line_when_title_matches_content(self) -> None:
        notifier = _build_schedule_notifier("cron", "喝水提醒：请记得喝水！")
        stream = io.StringIO()

        with redirect_stdout(stream):
            await notifier("喝水提醒：请记得喝水！")

        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        self.assertEqual(["[cron] 喝水提醒：请记得喝水！"], lines)


class CronModelRoundTripTests(unittest.TestCase):
    """CronJob / CronSchedule / CronPayload / CronJobState / CronStore の
    to_dict / from_dict ラウンドトリップ。
    """

    def test_cron_schedule_round_trip_at(self) -> None:
        original = CronSchedule(
            kind="at",
            at="2026-06-15T10:00:00+09:00",
        )
        data = original.to_dict()
        restored = CronSchedule.from_dict(data)
        self.assertEqual(original.kind, restored.kind)
        self.assertEqual(original.at, restored.at)
        self.assertIsNone(restored.every_seconds)
        self.assertIsNone(restored.expr)

    def test_cron_schedule_round_trip_every(self) -> None:
        original = CronSchedule(kind="every", every_seconds=300)
        data = original.to_dict()
        restored = CronSchedule.from_dict(data)
        self.assertEqual(original.kind, restored.kind)
        self.assertEqual(original.every_seconds, restored.every_seconds)

    def test_cron_schedule_round_trip_cron(self) -> None:
        original = CronSchedule(
            kind="cron",
            expr="0 9 * * 1-5",
            timezone="Asia/Tokyo",
        )
        data = original.to_dict()
        restored = CronSchedule.from_dict(data)
        self.assertEqual(original.kind, restored.kind)
        self.assertEqual(original.expr, restored.expr)
        self.assertEqual(original.timezone, restored.timezone)

    def test_cron_payload_round_trip(self) -> None:
        original = CronPayload(kind="text", content="hello", session_name="sess-1")
        data = original.to_dict()
        restored = CronPayload.from_dict(data)
        self.assertEqual(original.kind, restored.kind)
        self.assertEqual(original.content, restored.content)
        self.assertEqual(original.session_name, restored.session_name)

    def test_cron_job_state_round_trip(self) -> None:
        original = CronJobState(
            next_run_at="2026-06-15T10:00:00+09:00",
            last_run_at="2026-06-14T10:00:00+09:00",
            last_status="ok",
            last_error=None,
        )
        data = original.to_dict()
        restored = CronJobState.from_dict(data)
        self.assertEqual(original.next_run_at, restored.next_run_at)
        self.assertEqual(original.last_run_at, restored.last_run_at)
        self.assertEqual(original.last_status, restored.last_status)
        self.assertIsNone(restored.last_error)

    def test_cron_job_round_trip(self) -> None:
        schedule = CronSchedule(kind="every", every_seconds=60)
        payload = CronPayload(kind="agent", content="check", session_name="default")
        state = CronJobState(next_run_at="2026-06-15T10:00:00+09:00")
        original = CronJob(
            id="abc123",
            name="test-job",
            enabled=True,
            schedule=schedule,
            payload=payload,
            state=state,
            created_at="2026-06-15T09:00:00+09:00",
            updated_at="2026-06-15T09:00:00+09:00",
            delete_after_run=False,
        )
        data = original.to_dict()
        restored = CronJob.from_dict(data)
        self.assertEqual(original.id, restored.id)
        self.assertEqual(original.name, restored.name)
        self.assertEqual(original.enabled, restored.enabled)
        self.assertEqual(original.schedule.kind, restored.schedule.kind)
        self.assertEqual(original.payload.content, restored.payload.content)
        self.assertEqual(original.state.next_run_at, restored.state.next_run_at)
        self.assertEqual(original.created_at, restored.created_at)
        self.assertFalse(restored.delete_after_run)

    def test_cron_store_round_trip(self) -> None:
        job = CronJob(
            id="job-1",
            name="sample",
            schedule=CronSchedule(kind="every", every_seconds=120),
            payload=CronPayload(content="hello", session_name="sess"),
            created_at="2026-06-15T09:00:00+09:00",
            updated_at="2026-06-15T09:00:00+09:00",
        )
        original = CronStore(version=1, jobs=[job])
        data = original.to_dict()
        restored = CronStore.from_dict(data)
        self.assertEqual(original.version, restored.version)
        self.assertEqual(len(original.jobs), len(restored.jobs))
        self.assertEqual(original.jobs[0].id, restored.jobs[0].id)
        self.assertEqual(original.jobs[0].name, restored.jobs[0].name)


class CronParserEdgeCaseTests(unittest.TestCase):
    """normalize_schedule / describe_schedule のエッジケース。
    """

    def test_describe_schedule_at(self) -> None:
        desc = describe_schedule(
            CronSchedule(kind="at", at="2026-06-15T10:00:00+09:00")
        )
        self.assertEqual(desc, "at 2026-06-15T10:00:00+09:00")

    def test_describe_schedule_every(self) -> None:
        desc = describe_schedule(CronSchedule(kind="every", every_seconds=60))
        self.assertEqual(desc, "every 60s")

    def test_describe_schedule_cron(self) -> None:
        desc = describe_schedule(
            CronSchedule(kind="cron", expr="0 9 * * *")
        )
        self.assertEqual(desc, "cron 0 9 * * *")

    def test_describe_schedule_cron_with_timezone(self) -> None:
        desc = describe_schedule(
            CronSchedule(kind="cron", expr="0 9 * * *", timezone="Asia/Tokyo")
        )
        self.assertEqual(desc, "cron 0 9 * * * (Asia/Tokyo)")

    def test_normalize_schedule_at_missing_at_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_schedule(CronSchedule(kind="at", at=None))

    def test_normalize_schedule_every_missing_seconds_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_schedule(CronSchedule(kind="every", every_seconds=None))

    def test_normalize_schedule_every_zero_seconds_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_schedule(CronSchedule(kind="every", every_seconds=0))

    def test_normalize_schedule_cron_missing_expr_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_schedule(CronSchedule(kind="cron", expr=None))

    def test_normalize_schedule_cron_invalid_expr_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_schedule(CronSchedule(kind="cron", expr="0 9 * *"))  # 4 fields

    def test_normalize_schedule_unknown_kind_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_schedule(CronSchedule(kind="unknown"))  # type: ignore[arg-type]


class CronServiceMethodTests(unittest.IsolatedAsyncioTestCase):
    """CronService の個別メソッド: get_job / remove_job / set_enabled / run_job / status。
    """

    async def test_get_job_returns_none_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            result = await service.get_job("nonexistent")
            self.assertIsNone(result)

    async def test_get_job_returns_job_after_add(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            added = await service.add_job(
                name="my-job",
                schedule=CronSchedule(kind="every", every_seconds=60),
                payload=CronPayload(content="hello", session_name="sess"),
            )
            fetched = await service.get_job(added.id)
            self.assertIsNotNone(fetched)
            assert fetched is not None
            self.assertEqual(added.id, fetched.id)
            self.assertEqual(added.name, fetched.name)

    async def test_remove_job_returns_false_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            result = await service.remove_job("nonexistent")
            self.assertFalse(result)

    async def test_remove_job_removes_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store_path = Path(temp_dir) / "cron" / "jobs.json"
            service = CronService(store_path)
            added = await service.add_job(
                name="to-remove",
                schedule=CronSchedule(kind="every", every_seconds=60),
                payload=CronPayload(content="hello", session_name="sess"),
            )
            removed = await service.remove_job(added.id)
            self.assertTrue(removed)
            self.assertIsNone(await service.get_job(added.id))
            # 永続化確認
            saved = json.loads(store_path.read_text(encoding="utf-8"))
            self.assertEqual(0, len(saved["jobs"]))

    async def test_set_enabled_disables_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            added = await service.add_job(
                name="toggle",
                schedule=CronSchedule(kind="every", every_seconds=60),
                payload=CronPayload(content="hello", session_name="sess"),
            )
            disabled = await service.set_enabled(added.id, enabled=False)
            self.assertIsNotNone(disabled)
            assert disabled is not None
            self.assertFalse(disabled.enabled)
            self.assertIsNone(disabled.state.next_run_at)
            # list_jobs (include_disabled=False) に含まれない
            enabled_jobs = await service.list_jobs()
            self.assertNotIn(added.id, [j.id for j in enabled_jobs])

    async def test_set_enabled_returns_none_for_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            result = await service.set_enabled("nonexistent", enabled=False)
            self.assertIsNone(result)

    async def test_set_enabled_reenables_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            added = await service.add_job(
                name="re-enable",
                schedule=CronSchedule(kind="every", every_seconds=60),
                payload=CronPayload(content="hello", session_name="sess"),
            )
            await service.set_enabled(added.id, enabled=False)
            re_enabled = await service.set_enabled(added.id, enabled=True)
            self.assertIsNotNone(re_enabled)
            assert re_enabled is not None
            self.assertTrue(re_enabled.enabled)
            self.assertIsNotNone(re_enabled.state.next_run_at)

    async def test_run_job_executes_on_job_callback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            executed: list[str] = []

            async def on_job(job) -> str:
                executed.append(job.id)
                return "ok"

            service = CronService(
                Path(temp_dir) / "cron" / "jobs.json",
                on_job=on_job,
            )
            added = await service.add_job(
                name="run-test",
                schedule=CronSchedule(kind="every", every_seconds=60),
                payload=CronPayload(content="hello", session_name="sess"),
            )
            result = await service.run_job(added.id)
            self.assertTrue(result)
            self.assertEqual([added.id], executed)

    async def test_run_job_disabled_requires_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            added = await service.add_job(
                name="disabled-run",
                schedule=CronSchedule(kind="every", every_seconds=60),
                payload=CronPayload(content="hello", session_name="sess"),
            )
            await service.set_enabled(added.id, enabled=False)
            result = await service.run_job(added.id)
            self.assertFalse(result)
            # force=True なら実行できる
            forced = await service.run_job(added.id, force=True)
            self.assertTrue(forced)

    async def test_status_reports_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = CronService(Path(temp_dir) / "cron" / "jobs.json")
            status_before = await service.status()
            self.assertFalse(status_before["enabled"])
            self.assertEqual(0, status_before["jobs"])

            await service.start()
            try:
                await service.add_job(
                    name="status-job",
                    schedule=CronSchedule(kind="every", every_seconds=60),
                    payload=CronPayload(content="hello", session_name="sess"),
                )
                status_after = await service.status()
                self.assertTrue(status_after["enabled"])
                self.assertEqual(1, status_after["jobs"])
                self.assertIsNotNone(status_after["next_run_at"])
            finally:
                await service.stop()


class SummarizeJobTests(unittest.TestCase):
    def test_summarize_job_basic(self) -> None:
        job = CronJob(
            id="abc123",
            name="my-job",
            enabled=True,
            schedule=CronSchedule(kind="every", every_seconds=60),
            payload=CronPayload(kind="agent", content="hello", session_name="sess-1"),
            state=CronJobState(next_run_at="2026-06-15T10:00:00+09:00"),
            created_at="2026-06-15T09:00:00+09:00",
            updated_at="2026-06-15T09:00:00+09:00",
        )
        summary = summarize_job(job)
        self.assertEqual("abc123", summary["id"])
        self.assertEqual("my-job", summary["name"])
        self.assertTrue(summary["enabled"])
        self.assertEqual("every 60s", summary["schedule"])
        self.assertEqual("agent", summary["payload_kind"])
        self.assertEqual("sess-1", summary["session_name"])
        self.assertEqual("2026-06-15T10:00:00+09:00", summary["next_run_at"])


class HeartbeatUtilTests(unittest.IsolatedAsyncioTestCase):
    """Heartbeat モジュールのユーティリティ関数。
    """

    def test_default_template_exists(self) -> None:
        self.assertIn("HEARTBEAT.md", DEFAULT_HEARTBEAT_TEMPLATE)
        self.assertIn("<!--", DEFAULT_HEARTBEAT_TEMPLATE)

    def test_has_meaningful_content_returns_false_for_empty(self) -> None:
        self.assertFalse(has_meaningful_heartbeat_content(""))

    def test_has_meaningful_content_returns_false_for_only_comments(self) -> None:
        content = "<!-- only a comment -->\n<!-- another -->"
        self.assertFalse(has_meaningful_heartbeat_content(content))

    def test_has_meaningful_content_returns_false_for_only_headers(self) -> None:
        content = "# HEARTBEAT.md\n\n"
        self.assertFalse(has_meaningful_heartbeat_content(content))

    def test_has_meaningful_content_returns_true_for_task(self) -> None:
        content = "- [ ] Check todos\n"
        self.assertTrue(has_meaningful_heartbeat_content(content))

    def test_has_meaningful_content_strips_html_comments(self) -> None:
        content = "<!-- comment -->\n- [ ] Real task\n"
        self.assertTrue(has_meaningful_heartbeat_content(content))

    async def test_read_or_create_heartbeat_file_creates_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "HEARTBEAT.md"
            content = await read_or_create_heartbeat_file(path)
            self.assertEqual(content, DEFAULT_HEARTBEAT_TEMPLATE)
            self.assertTrue(path.exists())

    async def test_read_or_create_heartbeat_file_reads_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "HEARTBEAT.md"
            path.write_text("- [ ] Custom task\n", encoding="utf-8")
            content = await read_or_create_heartbeat_file(path)
            self.assertEqual(content, "- [ ] Custom task\n")

    async def test_write_heartbeat_file_writes_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "HEARTBEAT.md"
            await write_heartbeat_file(path, "- [ ] Written\n")
            self.assertTrue(path.exists())
            self.assertEqual(path.read_text(encoding="utf-8"), "- [ ] Written\n")


if __name__ == "__main__":
    unittest.main()
