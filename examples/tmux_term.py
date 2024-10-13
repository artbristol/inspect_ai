from logging import getLogger
from typing import List

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, bash, tool
from inspect_ai.util import sandbox

logger = getLogger(__name__)


@tool
def term_send_text() -> Tool:
    async def execute(text: List[str]) -> bool:
        """
        Sends text to the terminal.

        Args:
            text: list of text to send, per tmux manual

        Returns:
            True
        """
        await sandbox().exec(["tmux", "send-keys"] + text)
        return True

    return execute


@tool
def term_read() -> Tool:
    async def execute():
        """
        Reads the terminal window.

        Returns:
            Contents of the terminal window.
        """
        res = await sandbox().exec(["tmux", "capture-pane", "-p"])

        return res.stdout

    return execute


@task
def tmux_shell():
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
        sandbox="local",
    )


if __name__ == "__main__":
    eval(
        tmux_shell(),
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
