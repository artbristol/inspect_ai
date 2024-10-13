from pathlib import Path
from typing import Any, Dict, List, Tuple

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import bash
from inspect_ai.tool._tools._tmux_term import term_read, term_send_text
from inspect_ai.util import SandboxEnvironmentSpec


def tmux_shell(sandbox_definition: SandboxEnvironmentSpec) -> Task:
    return Task(
        dataset=[
            Sample(
                input="Just reply with Hello World",
                target="Hello World",
            )
        ],
        solver=[
            use_tools([bash(), term_send_text(), term_read()]),
            generate(),
            generate(),
        ],
        scorer=exact(),
        sandbox=sandbox_definition,
    )


def test_not_installed() -> None:
    res: List[EvalLog] = eval(
        tmux_shell(
            (
                "docker",
                str(
                    Path(__file__).resolve().parent
                    / "tmux_terminal_no_tmux_compose.yaml"
                ),
            )
        ),
        get_model(
            "mockllm/model",
            custom_outputs=[
                ModelOutput.for_tool_call(
                    "mockllm/model",
                    tool_name=term_send_text.__name__,
                    tool_arguments={"text": ["emacs -nw", "Enter"]},
                ),
                ModelOutput.for_tool_call(
                    "mockllm/model",
                    tool_name=term_send_text.__name__,
                    tool_arguments={"text": ["C-h", "r"]},
                ),
                ModelOutput.for_tool_call(
                    "mockllm/model",
                    tool_name=term_read.__name__,
                    tool_arguments={},
                ),
                ModelOutput.from_content("mockllm/model", content="wooza"),
                ModelOutput.from_content("mockllm/model", content="wooza"),
                ModelOutput.from_content("mockllm/model", content="wooza"),
            ],
        ),
    )
    assert res
    assert res[0]
    assert res[0].error
    assert (
        "Could not execute tmux in the sandbox, is it installed" in res[0].error.message
    )


def run_eval_return_tool_calls(
    tools_calls_mockllm: List[Tuple[str, Dict[str, Any]]],
) -> List[ChatMessageTool]:
    custom_outputs = [
        ModelOutput.for_tool_call("mockllm/model", tool_name, tool_arguments)
        for tool_name, tool_arguments in tools_calls_mockllm
    ]

    res: List[EvalLog] = eval(
        tmux_shell(
            (
                "docker",
                str(Path(__file__).resolve().parent / "tmux_terminal_compose.yaml"),
            )
        ),
        get_model(
            "mockllm/model",
            custom_outputs=custom_outputs
            + [
                ModelOutput.from_content("mockllm/model", content="wooza"),
                ModelOutput.from_content("mockllm/model", content="wooza"),
            ],
        ),
        sandbox_cleanup=False,
    )
    assert res
    assert res[0]
    assert res[0].error is None
    assert res[0].samples
    sample = res[0].samples[0]
    tool_calls = [x for x in sample.messages if x.role == "tool"]
    return tool_calls


def test_simple() -> None:
    tools_calls_mockllm = [
        (term_send_text.__name__, {"text": ["echo $(( 11 + 23 ))", "Enter"]}),
        (term_read.__name__, {}),
    ]

    tool_calls = run_eval_return_tool_calls(tools_calls_mockllm)

    assert str(11 + 23) in tool_calls[1].text


def test_long_running() -> None:
    tools_calls_mockllm = [
        (term_send_text.__name__, {"text": ["sleep 5; echo $(( 11 + 23 ))", "Enter"]}),
        (term_read.__name__, {}),
    ]

    tool_calls = run_eval_return_tool_calls(tools_calls_mockllm)

    assert str(11 + 23) in tool_calls[1].text
