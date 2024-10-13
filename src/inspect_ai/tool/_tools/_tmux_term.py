import asyncio
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

# TODO: write PS1="\a$PS1" into the shell to start,
# then use  tmux list-windows -F '#{window_bell_flag}:#{window_name}' | grep '^1'
# but this will require hackery for the *current* window to still be flagged
# Look at https://github.com/tmux/tmux/issues/2264#issuecomment-641112735 for the alert-bell hook
# OR always do stuff in not the current window?
# OR use https://github.com/rickstaa/tmux-notify ?
# in fact, window_activity_flag looks maybe even better?
# plan: wait for window_activity
# then wait using monitor-silence?


# TODO: FileNotFoundError for local sandbox exec of nonexistent command?
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
    async def execute() -> str:
        """
        Reads the terminal window.

        Returns:
            Contents of the terminal window.
        """
        await _ensure_tmux_started()
        res = await sandbox().exec(["tmux", "capture-pane", "-p"])

        return res.stdout

    return execute


async def _ensure_tmux_started() -> None:
    problem_starting = True
    try:
        check_tmux = await sandbox().exec(["tmux", "-V"])  # just check the version
        if check_tmux.returncode == 0:
            problem_starting = False
    except (PermissionError, FileNotFoundError) as ex:
        logger.debug(f"Permission error starting tmux: {ex=}")
    if problem_starting:
        raise Exception("Could not execute tmux in the sandbox, is it installed?")


@task
def tmux_shell() -> Task:
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
