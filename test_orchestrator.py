"""
tests/test_orchestrator.py — unit tests for AgentOrchestrator

The OpenAI client is fully mocked so no real API calls are made.
"""
import json
import pytest
import fakeredis.aioredis as fakeredis
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.orchestrator import AgentOrchestrator
from app.models.schemas import TaskStatus


def _make_llm_response(content: str, finish_reason: str = "stop", tool_calls=None):
    """Factory for mock OpenAI ChatCompletion responses."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_call(name: str, arguments: dict, call_id: str = "call_001"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@pytest.fixture(autouse=True)
async def mock_redis(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr("app.core.cache._redis_client", fake)
    yield fake


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_agent_direct_answer(mock_client):
    """LLM returns final answer with no tool calls."""
    mock_client.chat.completions.create.return_value = _make_llm_response(
        content="The answer is 42.", finish_reason="stop"
    )
    agent = AgentOrchestrator(llm_client=mock_client)
    result = await agent.run(task="What is the meaning of life?", use_cache=False)

    assert result.status == TaskStatus.COMPLETED
    assert result.result == "The answer is 42."
    assert result.total_steps == 1
    assert result.error is None


@pytest.mark.asyncio
async def test_agent_uses_search_tool(mock_client):
    """LLM calls search tool then produces final answer."""
    search_call = _make_tool_call("search", {"query": "laptop sales 2024", "num_results": 3})

    # Step 1: tool call
    mock_client.chat.completions.create.side_effect = [
        _make_llm_response("Searching for data...", finish_reason="tool_calls", tool_calls=[search_call]),
        _make_llm_response("Based on search results, laptop sales grew 8% in 2024.", finish_reason="stop"),
    ]

    with patch("app.tools.registry.dispatch_tools_concurrently", new=AsyncMock(return_value=[
        {"snippets": ["Laptop shipments rose 8%"], "_tool_name": "search", "_duration_ms": 50.0, "cached": False}
    ])):
        agent = AgentOrchestrator(llm_client=mock_client)
        result = await agent.run(task="How did laptop sales perform in 2024?", use_cache=False)

    assert result.status == TaskStatus.COMPLETED
    assert result.total_steps == 2
    assert len(result.steps[0].tool_calls) == 1
    assert result.steps[0].tool_calls[0].tool == "search"


@pytest.mark.asyncio
async def test_agent_uses_sql_tool(mock_client):
    """LLM calls sql_executor tool."""
    sql_call = _make_tool_call("sql_executor", {"query": "SELECT * FROM sales LIMIT 5"})

    mock_client.chat.completions.create.side_effect = [
        _make_llm_response("Querying database...", finish_reason="tool_calls", tool_calls=[sql_call]),
        _make_llm_response("The top 5 sales records show Laptop Pro as the highest value product.", finish_reason="stop"),
    ]

    with patch("app.tools.registry.dispatch_tools_concurrently", new=AsyncMock(return_value=[
        {"rows": [{"product": "Laptop Pro", "amount": 2499.99}], "row_count": 1,
         "columns": ["product", "amount"], "_tool_name": "sql_executor", "_duration_ms": 10.0, "cached": False}
    ])):
        agent = AgentOrchestrator(llm_client=mock_client)
        result = await agent.run(task="Show me the top 5 sales records", use_cache=False)

    assert result.status == TaskStatus.COMPLETED
    assert result.steps[0].tool_calls[0].tool == "sql_executor"


@pytest.mark.asyncio
async def test_agent_result_is_cached(mock_client):
    """Second call with same task returns cached result."""
    mock_client.chat.completions.create.return_value = _make_llm_response(
        content="Cached answer here.", finish_reason="stop"
    )
    agent = AgentOrchestrator(llm_client=mock_client)

    r1 = await agent.run(task="Repeat this task for caching", use_cache=True)
    assert r1.cached is False

    r2 = await agent.run(task="Repeat this task for caching", use_cache=True)
    assert r2.cached is True
    # LLM should only have been called once
    assert mock_client.chat.completions.create.call_count == 1


@pytest.mark.asyncio
async def test_agent_llm_error_sets_failed_status(mock_client):
    """LLM exception causes FAILED status."""
    mock_client.chat.completions.create.side_effect = Exception("OpenAI API rate limit")
    agent = AgentOrchestrator(llm_client=mock_client)
    result = await agent.run(task="This will fail due to LLM error", use_cache=False)

    assert result.status == TaskStatus.FAILED
    assert result.error is not None
    assert "LLM error" in result.error
