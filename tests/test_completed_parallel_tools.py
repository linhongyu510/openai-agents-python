from __future__ import annotations

import asyncio
from typing import Any

import pytest
from openai.types.responses.response_reasoning_item import ResponseReasoningItem

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunHooks,
    Runner,
    RunState,
    SQLiteSession,
    ToolGuardrailFunctionOutput,
    ToolInputGuardrailTripwireTriggered,
    ToolOutputGuardrailTripwireTriggered,
    UserError,
    function_tool,
    input_guardrail,
    output_guardrail,
    tool_input_guardrail,
    tool_output_guardrail,
)
from agents.testing import ScriptedModel, assistant_message, function_call


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "failure", ["input_guardrail", "output_guardrail", "exception", "cancelled"]
)
async def test_completed_parallel_tools_survive_sibling_failure(
    streaming: bool, failure: str
) -> None:
    completed = asyncio.Event()
    audit_completed = asyncio.Event()
    invocations: list[str] = []
    tool_error = (
        asyncio.CancelledError("email cancelled")
        if failure == "cancelled"
        else RuntimeError("email unavailable")
    )

    @function_tool
    async def create_ticket() -> str:
        await audit_completed.wait()
        invocations.append("ticket")
        return "ticket T-1"

    @function_tool
    async def record_audit() -> str:
        invocations.append("audit")
        return "audit A-1"

    @tool_input_guardrail
    async def block_email(data: Any) -> ToolGuardrailFunctionOutput:
        await completed.wait()
        return ToolGuardrailFunctionOutput.raise_exception(output_info="blocked")

    @tool_output_guardrail
    async def block_email_output(data: Any) -> ToolGuardrailFunctionOutput:
        return ToolGuardrailFunctionOutput.raise_exception(output_info="blocked")

    @function_tool(
        failure_error_function=None,
        tool_input_guardrails=[block_email] if failure == "input_guardrail" else [],
        tool_output_guardrails=[block_email_output] if failure == "output_guardrail" else [],
    )
    async def send_email() -> str:
        await completed.wait()
        if failure in {"exception", "cancelled"}:
            raise tool_error
        return "sent"

    class Hooks(RunHooks):
        async def on_tool_end(self, context, agent, tool, result) -> None:
            if tool.name == "record_audit":
                audit_completed.set()
            elif tool.name == "create_ticket":
                completed.set()

    model = ScriptedModel(
        [
            [
                ResponseReasoningItem(id="rs_email", type="reasoning", summary=[]),
                function_call("send_email", {}, call_id="call_email"),
                ResponseReasoningItem(id="rs_ticket", type="reasoning", summary=[]),
                function_call("create_ticket", {}, call_id="call_ticket"),
                function_call("record_audit", {}, call_id="call_audit"),
            ],
            [assistant_message("done")],
        ]
    )
    agent = Agent(name="support", model=model, tools=[send_email, create_ticket, record_audit])
    session = SQLiteSession("completed-tools")
    result = None
    streamed_outputs: list[Any] = []

    async def run() -> None:
        nonlocal result
        if streaming:
            result = Runner.run_streamed(agent, "Create a ticket", session=session, hooks=Hooks())
            async for event in result.stream_events():
                if event.type == "run_item_stream_event" and event.name == "tool_output":
                    streamed_outputs.append(event.item.to_input_item())
        else:
            await Runner.run(agent, "Create a ticket", session=session, hooks=Hooks())

    try:
        error_type = {
            "input_guardrail": ToolInputGuardrailTripwireTriggered,
            "output_guardrail": ToolOutputGuardrailTripwireTriggered,
            "exception": UserError,
            "cancelled": asyncio.CancelledError,
        }[failure]
        with pytest.raises(error_type) as raised:
            await asyncio.wait_for(run(), timeout=5)

        assert invocations == ["audit", "ticket"]
        saved = await session.get_items()
        assert [(item.get("type") or item.get("role"), item.get("call_id")) for item in saved] == [
            ("user", None),
            ("reasoning", None),
            ("function_call", "call_ticket"),
            ("function_call", "call_audit"),
            ("function_call_output", "call_ticket"),
            ("function_call_output", "call_audit"),
        ]
        assert saved[1]["id"] == "rs_ticket"
        assert [item["output"] for item in saved[-2:]] == ["ticket T-1", "audit A-1"]
        if failure == "exception":
            assert raised.value.__cause__ is tool_error
        if not isinstance(raised.value, asyncio.CancelledError):
            assert raised.value.run_data is not None
            assert [item.type for item in raised.value.run_data.new_items] == [
                "reasoning_item",
                "tool_call_item",
                "tool_call_item",
                "tool_call_output_item",
                "tool_call_output_item",
            ]
        if streaming:
            assert result is not None
            assert streamed_outputs == saved[-2:]
            assert result.to_input_list()[1:] == saved[1:]
            restored = await RunState.from_json(agent, result.to_state().to_json())
            resumed = await Runner.run(agent, restored)
            assert resumed.final_output == "done"
            assert model.last_call is not None
            assert model.last_call.input == saved

        else:
            await Runner.run(agent, "Continue", session=session)
            assert model.last_call is not None
            assert model.last_call.input[:-1] == saved
        assert invocations == ["audit", "ticket"]
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_parent_cancellation_does_not_recover_a_partial_turn(streaming: bool) -> None:
    completed = asyncio.Event()
    pending_started = asyncio.Event()
    pending_stopped = asyncio.Event()

    @function_tool
    async def create_ticket() -> str:
        await pending_started.wait()
        return "ticket T-1"

    @function_tool(failure_error_function=None)
    async def pending() -> str:
        pending_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            pending_stopped.set()
        return "unreachable"

    class Hooks(RunHooks):
        async def on_tool_end(self, context, agent, tool, result) -> None:
            completed.set()

    agent = Agent(
        name="support",
        model=ScriptedModel(
            [
                [
                    function_call("create_ticket", {}, call_id="ticket"),
                    function_call("pending", {}, call_id="pending"),
                ]
            ]
        ),
        tools=[create_ticket, pending],
    )
    session = SQLiteSession("parent-cancelled")

    async def run() -> None:
        if streaming:
            result = Runner.run_streamed(agent, "Create a ticket", session=session, hooks=Hooks())
            async for _ in result.stream_events():
                pass
        else:
            await Runner.run(agent, "Create a ticket", session=session, hooks=Hooks())

    task = asyncio.create_task(run())
    try:
        await asyncio.wait_for(completed.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        await asyncio.wait_for(pending_stopped.wait(), timeout=5)
        assert all(item.get("type") != "function_call_output" for item in await session.get_items())
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_completed_tool_survives_its_end_hook_failure(streaming: bool) -> None:
    calls = 0
    error = UserError("hook failed")

    @function_tool
    async def create_ticket() -> str:
        nonlocal calls
        calls += 1
        return "ticket T-1"

    class Hooks(RunHooks):
        async def on_tool_end(self, context, agent, tool, result) -> None:
            raise error

    agent = Agent(
        name="support",
        model=ScriptedModel([[function_call("create_ticket", {}, call_id="ticket")]]),
        tools=[create_ticket],
    )
    session = SQLiteSession("failed-hook")
    try:
        with pytest.raises(UserError) as raised:
            if streaming:
                result = Runner.run_streamed(
                    agent, "Create a ticket", hooks=Hooks(), session=session
                )
                async for _ in result.stream_events():
                    pass
                pytest.fail("Expected the hook failure")
            else:
                await Runner.run(agent, "Create a ticket", hooks=Hooks(), session=session)
        assert raised.value is error
        assert calls == 1
        saved = await session.get_items()
        assert [item.get("type") or item.get("role") for item in saved] == [
            "user",
            "function_call",
            "function_call_output",
        ]
        assert saved[-1]["output"] == "ticket T-1"
    finally:
        session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("gate", ["input", "terminal_output"])
async def test_failed_turn_does_not_publish_unadmitted_tool_outputs(
    streaming: bool, gate: str
) -> None:
    completed = asyncio.Event()
    calls: list[str] = []

    @function_tool
    async def create_ticket() -> str:
        calls.append("ticket")
        return "ticket T-1"

    @function_tool(failure_error_function=None)
    async def fail() -> str:
        await completed.wait()
        raise RuntimeError("sibling failed")

    class Hooks(RunHooks):
        async def on_tool_end(self, context, agent, tool, result) -> None:
            completed.set()

    @input_guardrail
    async def reject_input(ctx, agent, input) -> GuardrailFunctionOutput:
        await completed.wait()
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    @output_guardrail
    async def reject_output(ctx, agent, output) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    agent = Agent(
        name="support",
        model=ScriptedModel(
            [
                [
                    function_call("create_ticket", {}, call_id="ticket"),
                    function_call("fail", {}, call_id="failure"),
                ]
            ]
        ),
        tools=[create_ticket, fail],
        input_guardrails=[reject_input] if gate == "input" else [],
        output_guardrails=[reject_output] if gate == "terminal_output" else [],
        tool_use_behavior="stop_on_first_tool" if gate == "terminal_output" else "run_llm_again",
    )
    session = SQLiteSession("guarded-output")
    result = None

    async def run() -> None:
        nonlocal result
        if streaming:
            result = Runner.run_streamed(agent, "Create a ticket", session=session, hooks=Hooks())
            async for event in result.stream_events():
                assert event.type != "run_item_stream_event" or event.name != "tool_output"
        else:
            await Runner.run(agent, "Create a ticket", session=session, hooks=Hooks())

    try:
        with pytest.raises(InputGuardrailTripwireTriggered if gate == "input" else UserError):
            await asyncio.wait_for(run(), timeout=5)
        assert calls == ["ticket"]
        assert all(item.get("type") != "function_call_output" for item in await session.get_items())
        if result is not None:
            assert not result.new_items
            assert all(
                item.get("type") != "function_call_output" for item in result.to_input_list()
            )
    finally:
        session.close()
