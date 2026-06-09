from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path (like conftest.py does for pytest)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
project_root_text = str(PROJECT_ROOT)
if project_root_text not in sys.path:
    sys.path.insert(0, project_root_text)

import unittest

from amadeus.models import LLMMessage, LLMResponse
from amadeus.orchestration.decision import (
    DECISION_SYSTEM_PROMPT,
    DecisionEngine,
    _rule_based_decision,
)
from amadeus.agent import AgentCore
from amadeus.providers.base import LLMProvider


class FakeDecisionProvider(LLMProvider):
    """Fake LLM that returns a controlled route decision."""

    def __init__(self, route: str = "chat", reason: str = "test"):
        self.route = route
        self.reason = reason
        self.last_messages: list[LLMMessage] = []

    async def generate(self, messages, *, tools=None, tool_choice=None,
                       temperature=None, max_tokens=None) -> LLMResponse:
        self.last_messages = list(messages)
        return LLMResponse(
            message=LLMMessage(
                role="assistant",
                content=f'{{"route":"{self.route}","reason":"{self.reason}"}}',
            ),
            model="fake-decision",
        )


class TestDecisionRuleBased(unittest.TestCase):
    """Tests for rule-based decision patterns (no LLM needed)."""

    def test_chinese_agent_patterns(self):
        agent_inputs = [
            "帮我写一个爬虫脚本",
            "帮我查看项目代码",
            "搜索仓库中的bug",
            "删除测试文件",
            "创建新模块",
            "运行测试",
            "记住这个偏好",
            "查一下记忆",
            "设置提醒半小时后开会",
            "30秒后提醒我喝水",
            "每天提醒我备份",
            "开启心跳任务",
            "检查cron状态",
        ]
        for text in agent_inputs:
            with self.subTest(input=text):
                decision = _rule_based_decision(text)
                self.assertIsNotNone(decision,
                    f"Expected rule match for: {text}")
                self.assertEqual(decision.route, "agent")

    def test_english_agent_patterns(self):
        agent_inputs = [
            "write a script",
            "open the file main.py",
            "search the codebase for bugs",
            "edit the file",
            "create a module",
            "remember this preference",
            "set a reminder in 30 minutes",
            "check the cron status",
            "enable heartbeat",
        ]
        for text in agent_inputs:
            with self.subTest(input=text):
                decision = _rule_based_decision(text)
                self.assertIsNotNone(decision,
                    f"Expected rule match for: {text}")
                self.assertEqual(decision.route, "agent")

    def test_chat_no_rule_match(self):
        chat_inputs = [
            "你好",
            "今天天气不错",
            "讲个笑话",
            "你喜欢什么颜色",
            "谢谢你的帮助",
            "晚安",
        ]
        for text in chat_inputs:
            with self.subTest(input=text):
                decision = _rule_based_decision(text)
                self.assertIsNone(decision,
                    f"Expected no rule match for chat: {text}")

    def test_empty_returns_chat_decision(self):
        # Empty input returns a RouteDecision(chat), not None
        from amadeus.orchestration.decision import RouteDecision
        d = _rule_based_decision("")
        self.assertIsNotNone(d)
        self.assertEqual(d.route, "chat")
        d2 = _rule_based_decision("   ")
        self.assertEqual(d2.route, "chat")

    def test_route_decision_properties(self):
        from amadeus.orchestration.decision import RouteDecision
        d = RouteDecision(route="agent", reason="test")
        self.assertTrue(d.requires_agent)
        d2 = RouteDecision(route="chat", reason="test")
        self.assertFalse(d2.requires_agent)


class TestDecisionEngine(unittest.TestCase):
    """Tests for DecisionEngine with route modes and LLM decider."""

    def test_chat_only_override(self):
        engine = DecisionEngine()
        decision = _run_async(engine.decide(
            "帮我写代码", route_mode="chat_only"))
        self.assertEqual(decision.route, "chat")

    def test_force_agent_override(self):
        engine = DecisionEngine()
        decision = _run_async(engine.decide(
            "你好啊", route_mode="force_agent"))
        self.assertEqual(decision.route, "agent")

    def test_fallback_to_chat_without_llm(self):
        engine = DecisionEngine()
        decision = _run_async(engine.decide(
            "some ambiguous message with no rule match"))
        self.assertEqual(decision.route, "chat")

    def test_llm_decider_used_when_available(self):
        provider = FakeDecisionProvider(route="agent", reason="LLM says so")
        agent = AgentCore(provider, system_prompt=DECISION_SYSTEM_PROMPT)
        engine = DecisionEngine(agent)
        decision = _run_async(engine.decide(
            "should I use tools for this?"))
        self.assertEqual(decision.route, "agent")


def _run_async(coro):
    import asyncio
    import concurrent.futures
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, coro)
            return future.result(timeout=10)
    except RuntimeError:
        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
