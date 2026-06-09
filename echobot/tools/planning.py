from __future__ import annotations

from typing import Any

from .base import (
    BaseTool,
    ToolExecutionOutput,
    ToolLoopControl,
    ToolOutput,
    ToolTraceEvent,
)


_PLAN_STATUSES = {"pending", "in_progress", "completed"}


class UpdatePlanTool(BaseTool):
    name = "update_plan"
    description = (
        "Record or update a short plan for the current task. "
        "Use this for multi-step work and keep the statuses current."
    )
    parameters = {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "description": "Optional short explanation for the plan update.",
                "default": "",
            },
            "plan": {
                "type": "array",
                "description": "The current plan.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {
                            "type": "string",
                            "description": "A short step description.",
                        },
                        "status": {
                            "type": "string",
                            "description": "One of pending, in_progress, completed.",
                            "enum": ["pending", "in_progress", "completed"],
                        },
                    },
                    "required": ["step", "status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["plan"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self._latest_plan: list[dict[str, str]] = []

    async def run(self, arguments: dict[str, Any]) -> ToolOutput:
        explanation = str(arguments.get("explanation", "")).strip()
        plan = _normalize_plan(arguments.get("plan"))
        self._latest_plan = plan
        current_step = _current_plan_step(plan)

        details_lines = []
        if explanation:
            details_lines.append(explanation)
            details_lines.append("")
        for item in plan:
            details_lines.append(f"[{item['status']}] {item['step']}")

        return ToolExecutionOutput(
            data={
                "kind": "plan_update",
                "explanation": explanation,
                "plan": plan,
                "current_step": current_step,
            },
            trace_events=[
                ToolTraceEvent(
                    event="plan_updated",
                    data={
                        "title": "Plan updated",
                        "summary": _plan_summary(plan),
                        "details": "\n".join(details_lines).strip(),
                        "explanation": explanation,
                        "plan": plan,
                        "current_step": current_step,
                    },
                )
            ],
        )


class RequestUserInputTool(BaseTool):
    name = "request_user_input"
    description = (
        "Pause the current task and ask the user a focused follow-up question "
        "when you are blocked by missing information."
    )
    parameters = {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The exact short question or request the user should see.",
            },
            "choices": {
                "type": "array",
                "description": "Optional short answer choices to help the user reply.",
                "items": {
                    "type": "string",
                },
                "default": [],
            },
            "why_needed": {
                "type": "string",
                "description": "Optional short internal note explaining why this answer is needed.",
                "default": "",
            },
        },
        "required": ["prompt"],
        "additionalProperties": False,
    }

    async def run(self, arguments: dict[str, Any]) -> ToolOutput:
        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("prompt is required")

        raw_choices = arguments.get("choices", [])
        if raw_choices is None:
            raw_choices = []
        if not isinstance(raw_choices, list):
            raise ValueError("choices must be an array of strings")
        choices = [str(choice).strip() for choice in raw_choices if str(choice).strip()]
        why_needed = str(arguments.get("why_needed", "")).strip()

        response_lines = [prompt]
        if choices:
            response_lines.append("")
            response_lines.append("可参考的回答：")
            response_lines.extend(f"- {choice}" for choice in choices)

        response_text = "\n".join(response_lines).strip()
        pending_request = {
            "prompt": prompt,
            "choices": choices,
            "why_needed": why_needed,
        }

        return ToolExecutionOutput(
            data={
                "kind": "user_input_request",
                "request": pending_request,
            },
            trace_events=[
                ToolTraceEvent(
                    event="user_input_requested",
                    data={
                        "title": "Waiting for user input",
                        "summary": prompt,
                        "details": _user_input_details(prompt, choices, why_needed),
                        "request": pending_request,
                    },
                )
            ],
            control=ToolLoopControl(
                action="await_user_input",
                status="waiting_for_input",
                response_content=response_text,
                metadata=pending_request,
            ),
        )


def _normalize_plan(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("plan must be a non-empty array")

    normalized_plan: list[dict[str, str]] = []
    in_progress_count = 0

    for item in value:
        if not isinstance(item, dict):
            raise ValueError("each plan item must be an object")

        step = str(item.get("step", "")).strip()
        status = str(item.get("status", "")).strip()
        if not step:
            raise ValueError("each plan item must include a non-empty step")
        if status not in _PLAN_STATUSES:
            raise ValueError("plan status must be pending, in_progress, or completed")
        if status == "in_progress":
            in_progress_count += 1
        normalized_plan.append(
            {
                "step": step,
                "status": status,
            }
        )

    if in_progress_count > 1:
        raise ValueError("only one plan item can be in_progress")

    return normalized_plan


def _current_plan_step(plan: list[dict[str, str]]) -> str:
    for item in plan:
        if item["status"] == "in_progress":
            return item["step"]
    return ""


def _plan_summary(plan: list[dict[str, str]]) -> str:
    current_step = _current_plan_step(plan)
    if current_step:
        return f"{len(plan)} steps, current: {current_step}"
    return f"{len(plan)} steps"


def _user_input_details(
    prompt: str,
    choices: list[str],
    why_needed: str,
) -> str:
    lines = [prompt]
    if choices:
        lines.append("")
        lines.append("Choices:")
        lines.extend(f"- {choice}" for choice in choices)
    if why_needed:
        lines.append("")
        lines.append("Why needed:")
        lines.append(why_needed)
    return "\n".join(lines).strip()
