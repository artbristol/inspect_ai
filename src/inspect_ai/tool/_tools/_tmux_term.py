from logging import getLogger
from typing import List

from inspect_ai.tool import Tool, tool
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

tmux_session_name = "_tmux_term.py_session"


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
        await _ensure_tmux_started()
        send_text_res = await sandbox().exec(["tmux", "send-keys"] + text)
        if send_text_res.returncode != 0:
            raise Exception(f"problem sending text: {send_text_res=}")
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
        read_res = await sandbox().exec(["tmux", "capture-pane", "-p"])
        if read_res.returncode != 0:
            raise Exception(f"problem reading: {read_res=}")

        return read_res.stdout

    return execute


async def _ensure_tmux_started() -> None:
    tmux_execution_problem = True

    try:
        check_tmux = await sandbox().exec(["tmux", "-V"])
        if check_tmux.returncode == 0:
            tmux_execution_problem = False
    except (PermissionError, FileNotFoundError) as ex:
        logger.debug(f"Permission error starting tmux: {ex=}")

    if tmux_execution_problem:
        raise Exception("Could not execute tmux in the sandbox, is it installed?")

    check_tmux = await sandbox().exec(
        ["tmux", "new-session", "-d", "-t", tmux_session_name]
    )
    if check_tmux.returncode != 0:
        if check_tmux.returncode == 1 and "duplicate session" in check_tmux.stderr:
            logger.debug("tmux already started")
        else:
            raise Exception(f"problem starting session: {check_tmux=}")
