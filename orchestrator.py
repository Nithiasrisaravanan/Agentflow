"""
app/agents/orchestrator.py — ReAct-style LLM agent with async orchestration.

The agent iterates through Thought → Tool-call(s) → Observation cycles
until the LLM produces a final answer or MAX_AGENT_STEPS is reached.

All intermediate results are cached in Redis so retries are cheap.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from app.core.cache import cache_get, cache_set
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import AgentStep, TaskResponse, TaskStatus, ToolCall, ToolName
from app.tools.definitions import TOOL_DEFINITIONS
from app.tools.registry import dispatch_tools_concurrently

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are AgentFlow, an intelligent task-automation assistant.

You have access to the following tools:
- summarizer  : summarise long text
- sql_executor: run SELECT queries on the sales database
- search      : look up current information

Work step by step:
1. Think about what you need to do.
2. Call one or more tools if needed.
3. Use the tool results to progress toward the final answer.
4. When you have enough information, respond with a clear final answer.

Always be concise and factual. If a tool returns an error, try an alternative approach.
"""


class AgentOrchestrator:
    """Async ReAct-style agent orchestrator."""

    def __init__(self, llm_client: Optional[AsyncOpenAI] = None) -> None:
        self.settings = get_settings()
        self.client = llm_client or AsyncOpenAI(api_key=self.settings.openai_api_key)

    # ── Public entry-point ────────────────────────────────────────────────────

    async def run(self, task: str, context: Optional[Dict[str, Any]] = None, use_cache: bool = True) -> TaskResponse:
        task_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()
        logger.info("Agent task started", task_id=task_id, task=task[:120])

        # Check top-level result cache
        cache_payload = {"task": task, "context": context}
        if use_cache:
            cached = await cache_get("agent_result", cache_payload)
            if cached:
                logger.info("Agent result cache hit", task_id=task_id)
                return TaskResponse(**{**cached, "task_id": task_id, "cached": True})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._build_user_message(task, context)},
        ]
        steps: List[AgentStep] = []
        final_answer: Optional[str] = None
        error: Optional[str] = None

        for step_num in range(1, self.settings.max_agent_steps + 1):
            logger.debug("Agent step", task_id=task_id, step=step_num)

            try:
                response = await self.client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                    tool_choice="auto",
                    max_tokens=self.settings.openai_max_tokens,
                    temperature=self.settings.openai_temperature,
                )
            except Exception as exc:
                logger.exception("LLM call failed", task_id=task_id, step=step_num, error=str(exc))
                error = f"LLM error at step {step_num}: {exc}"
                break

            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Collect thought
            thought = msg.content or ""

            # Build step record
            step = AgentStep(step=step_num, thought=thought)

            # ── No tool calls → final answer ──────────────────────────────────
            if finish_reason == "stop" or not msg.tool_calls:
                final_answer = thought
                steps.append(step)
                break

            # ── Dispatch tool calls concurrently ──────────────────────────────
            raw_calls = [
                {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments or "{}"),
                    "_id": tc.id,
                }
                for tc in msg.tool_calls
            ]

            tool_results = await dispatch_tools_concurrently(raw_calls, llm_client=self.client)

            # Map results back to OpenAI tool-result messages
            for tc, result in zip(msg.tool_calls, tool_results):
                result_str = json.dumps(result, default=str)

                # Record in step
                step.tool_calls.append(
                    ToolCall(
                        tool=ToolName(tc.function.name),
                        input=json.loads(tc.function.arguments or "{}"),
                        output=result,
                        duration_ms=result.get("_duration_ms"),
                    )
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

            # Append assistant message with tool calls
            messages.append(msg)  # type: ignore[arg-type]
            step.intermediate_result = f"{len(tool_results)} tool(s) executed"
            steps.append(step)

        else:
            error = f"Agent exceeded maximum steps ({self.settings.max_agent_steps})"
            logger.warning("Agent max steps reached", task_id=task_id)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        status = TaskStatus.COMPLETED if final_answer else TaskStatus.FAILED

        response_data = {
            "task_id": task_id,
            "status": status,
            "result": final_answer,
            "steps": [s.model_dump() for s in steps],
            "total_steps": len(steps),
            "duration_ms": duration_ms,
            "cached": False,
            "error": error,
        }

        if use_cache and final_answer:
            await cache_set("agent_result", cache_payload, response_data)

        logger.info(
            "Agent task finished",
            task_id=task_id,
            status=status,
            steps=len(steps),
            duration_ms=duration_ms,
        )
        return TaskResponse(**response_data)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_user_message(task: str, context: Optional[Dict[str, Any]]) -> str:
        if context:
            ctx_str = "\n".join(f"  {k}: {v}" for k, v in context.items())
            return f"Task: {task}\n\nAdditional context:\n{ctx_str}"
        return f"Task: {task}"
