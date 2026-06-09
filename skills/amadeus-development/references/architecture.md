# Architecture

Read this file when a change crosses module boundaries or when you need to trace a user turn end to end.

## Top-level entrypoints

- `amadeus/cli/main.py` is the unified CLI. With no subcommand it falls back to chat mode.
- `amadeus/cli/chat.py`, `amadeus/cli/gateway.py`, and `amadeus/cli/app.py` all build the shared runtime through `amadeus/runtime/bootstrap.py`.
- `amadeus/app/create_app.py` builds the FastAPI app, mounts routers, and serves the browser UI from `amadeus/app/web/`.
- `amadeus/app/runtime.py` wraps the shared runtime with channel, gateway, ASR, TTS, delivery, and web-console services.

## Shared runtime assembly

`build_runtime_context(...)` in `amadeus/runtime/bootstrap.py` is the single assembly point. It creates:

- `AttachmentStore`
- provider instances for the main agent, decider, and roleplay layers
- `AgentCore`
- session stores for user sessions and agent sessions
- `AgentTraceStore`
- a `ToolRegistry` factory
- `SkillRegistry`
- `CronService` and optional `HeartbeatService`
- `SessionAgentRunner`
- `RoleCardRegistry`
- `DecisionEngine`
- `RoleplayEngine`
- `ConversationCoordinator`

If a feature should exist in chat, gateway, and app, wire it here once.

## User turn flow

1. A CLI command, gateway event, or API handler resolves a session and calls the coordinator.
2. `ConversationCoordinator.handle_user_turn_stream(...)` loads session state, role, and route mode.
3. Route mode decides how much routing logic to run: `chat_only` and `force_agent` short-circuit, while `auto` calls `DecisionEngine.decide(...)`.
4. Chat turns go to `RoleplayEngine.stream_chat_reply(...)`.
5. Agent turns can emit a delegated acknowledgement through the roleplay layer, then start background work.
6. `SessionAgentRunner.run_prompt(...)` calls `run_agent_turn(...)`.
7. `run_agent_turn(...)` prefers `ask_with_skills(...)`, then `ask_with_tools(...)`, then `ask_with_memory(...)`.
8. `AgentCore.ask_with_skills(...)` adds the skill catalog prompt, injects explicit `/skill-name` or `$skill-name` activations, and layers in `activate_skill`, `list_skill_resources`, and `read_skill_resource`.
9. Final agent output flows back through `RoleplayEngine.present_agent_result(...)`, `present_agent_failure(...)`, or scheduled-task presenters before it reaches CLI, gateway, or API clients.

## Base tools and skills

- `create_basic_tool_registry(...)` in `amadeus/tools/builtin.py` builds the shared base tool set.
- The default tools are current time, directory listing, text file read or write, web requests, and shell commands.
- If attachments are enabled, media tools are added for viewing images and sending image or file outputs back to the user.
- If memory is enabled, the memory search tool is added.
- If cron is enabled, the cron tool is added. Scheduled runs keep the tool readable but disable mutations.
- Skill tools are not part of the base registry. They are layered in by `AgentCore.ask_with_skills(...)`.

## Layer boundaries

### Decision layer

- `amadeus/orchestration/decision.py`
- `amadeus/orchestration/route_modes.py`
- Use rules for obvious tool, workspace, memory, and scheduling requests.
- Use the lightweight decider LLM only for ambiguous turns.

### Roleplay layer

- `amadeus/orchestration/roleplay.py`
- Use only visible conversation context plus role-card and system instructions.
- Do not inspect files, tools, schedules, or memory directly.

### Agent layer

- `amadeus/agent.py`
- `amadeus/runtime/session_runner.py`
- `amadeus/runtime/turns.py`
- Own tool use, skill use, file access, shell calls, memory lookup, scheduling changes, and attachment-aware execution.

### Commands and control surfaces

- `amadeus/commands/` handles `/route`, `/runtime`, `/role`, and saved-session command parsing and execution.
- `amadeus/cli/session_commands.py` handles `/session ...` inside the interactive CLI chat loop.
- `amadeus/app/routers/` exposes HTTP endpoints for chat, sessions, roles, cron, heartbeat, channels, attachments, and web-console assets.

## Skills and role cards

- `SkillRegistry.discover(...)` searches project skills first, then local managed roots, then built-in skills.
- Project skills under `skills/` override built-in skills with the same `name`.
- Role cards are discovered from `amadeus/roles/`, `roles/`, and `.amadeus/roles/`.
- `.amadeus/roles/default.md` is created automatically when missing.
- This repository can rely entirely on `roles/` or `.amadeus/roles/`; `amadeus/roles/` is an optional built-in root and may be absent.

## State files and managed directories

- Sessions: `.amadeus/sessions/`
- Agent-side session history: `.amadeus/agent_sessions/`
- Agent traces: `.amadeus/agent_traces/`
- Runtime settings: `.amadeus/runtime_settings.json`
- Cron store: `.amadeus/cron/jobs.json`
- Heartbeat file: `.amadeus/HEARTBEAT.md`
- Channel config: `.amadeus/channels.json`
- Delivery state: `.amadeus/delivery.json`
- Gateway route sessions: `.amadeus/route_sessions.json`
- Attachments: `.amadeus/attachments/`
- Managed roles: `.amadeus/roles/`
- Live2D uploads: `.amadeus/live2d/`
- Uploaded stage backgrounds: `.amadeus/stage_backgrounds/`

## Current module map

- `amadeus/skill_support/`: skill discovery, parsing, explicit activation, and lazy resource tools
- `amadeus/tools/`: shared tool registry plus filesystem, shell, web, media, memory, and cron tools
- `amadeus/runtime/`: bootstrap, sessions, traces, turn execution, and runtime settings
- `amadeus/orchestration/`: coordinator, decision, roleplay, jobs, route modes, and role-card registry
- `amadeus/commands/`: command parsing and execution for route mode, runtime settings, roles, and saved sessions
- `amadeus/channels/`: channel configs, message bus, manager, registry, and platform adapters
- `amadeus/gateway/`: route-session mapping, delivery state, and gateway runtime wiring
- `amadeus/app/services/web_console/`: Live2D, stage, and runtime-settings helpers for the browser UI
- `amadeus/app/web/features/`: browser-side features grouped by chat, sessions, layout, Live2D, TTS, and ASR
- `amadeus/attachments.py` and `amadeus/images.py`: attachment persistence plus image limits and promotion
- `amadeus/memory/`: ReMeLight integration
- `amadeus/asr/` and `amadeus/tts/`: speech services

## Test map

- `tests/test_skill_support.py`: skill discovery, activation, parsing, and lazy resource loading
- `tests/test_chat_agent.py`: CLI trace labels and async chat-loop behavior
- `tests/test_agent.py`, `tests/test_tools.py`, and `tests/test_agent_traces.py`: agent loop, tool execution, and trace persistence
- `tests/test_decision.py`, `tests/test_roleplay.py`, `tests/test_coordinator.py`, and `tests/test_roles.py`: routing, presentation, coordinator behavior, and role cards
- `tests/test_commands.py`, `tests/test_gateway.py`, and `tests/test_app_api.py`: command, gateway, and API surfaces
- `tests/test_sessions.py`, `tests/test_config.py`, and `tests/test_scheduler.py`: persisted runtime state and runtime configuration
- `tests/test_images.py`, `tests/test_channel_images.py`, and `tests/test_tts.py`: media-related flows
