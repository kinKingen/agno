from typing import Any, AsyncIterator, Iterator

import pytest

from agno.agent import Agent
from agno.exceptions import RetryAgentRun, StopAgentRun
from agno.models.base import Model
from agno.models.response import ModelResponse

STOP_MESSAGE = "Value exceeds threshold. Stopping tool call execution."


class StopToolModel(Model):
    """Offline model that always calls the tool which stops the run."""

    def _response(self) -> ModelResponse:
        return ModelResponse(
            role="assistant",
            tool_calls=[
                {
                    "id": "call-stop",
                    "type": "function",
                    "function": {"name": "stop_now", "arguments": "{}"},
                }
            ],
        )

    def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._response()

    async def ainvoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
        return self._response()

    def invoke_stream(self, *args: Any, **kwargs: Any) -> Iterator[ModelResponse]:
        yield self._response()

    async def ainvoke_stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ModelResponse]:
        yield self._response()

    def parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response

    def _parse_provider_response(self, response: Any, **kwargs: Any) -> ModelResponse:
        return response

    def _parse_provider_response_delta(self, response: Any) -> ModelResponse:
        return response


class RetryToolModel(StopToolModel):
    """Offline model that recovers after a retryable tool exception."""

    def __init__(self, id: str):
        super().__init__(id=id)
        self.response_count = 0

    def _response(self) -> ModelResponse:
        self.response_count += 1
        if self.response_count == 1:
            return ModelResponse(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call-retry",
                        "type": "function",
                        "function": {"name": "retry_now", "arguments": "{}"},
                    }
                ],
            )
        return ModelResponse(role="assistant", content="Recovered after retry")


def stop_now() -> str:
    raise StopAgentRun("stop", agent_message=STOP_MESSAGE)


def retry_now() -> str:
    raise RetryAgentRun("retry", agent_message="Retrying with more context")


def _build_agent() -> Agent:
    return Agent(model=StopToolModel(id="stop-tool-model"), tools=[stop_now], telemetry=False)


def test_stop_agent_run_agent_message_is_returned_as_response_content():
    response = _build_agent().run("Stop the run")

    assert response.content == STOP_MESSAGE


@pytest.mark.asyncio
async def test_stop_agent_run_agent_message_is_returned_as_async_response_content():
    response = await _build_agent().arun("Stop the run")

    assert response.content == STOP_MESSAGE


def test_stop_agent_run_agent_message_is_streamed():
    events = list(_build_agent().run("Stop the run", stream=True))

    assert [event.content for event in events] == [STOP_MESSAGE]


@pytest.mark.asyncio
async def test_stop_agent_run_agent_message_is_streamed_async():
    events = [event async for event in _build_agent().arun("Stop the run", stream=True)]

    assert [event.content for event in events] == [STOP_MESSAGE]


def test_retry_agent_run_agent_message_remains_model_context_only():
    agent = Agent(model=RetryToolModel(id="retry-tool-model"), tools=[retry_now], telemetry=False)

    response = agent.run("Retry the tool")

    assert response.content == "Recovered after retry"


@pytest.mark.asyncio
async def test_retry_agent_run_agent_message_remains_model_context_only_async():
    agent = Agent(model=RetryToolModel(id="retry-tool-model"), tools=[retry_now], telemetry=False)

    response = await agent.arun("Retry the tool")

    assert response.content == "Recovered after retry"
