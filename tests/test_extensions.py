# type: ignore

import subprocess
import sys

import pytest
from test_helpers.tools import list_files

from inspect_ai import Task, eval_async
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate, use_tools


@pytest.mark.asyncio
async def test_extension_model():
    # ensure the package is installed
    ensure_package_installed()

    # call the model
    mdl = get_model("custom/gpt7")
    result = await mdl.generate({"role": "user", "content": "hello"}, [], "none", {})
    assert result.completion == "Hello from gpt7"


@pytest.mark.asyncio
async def test_extension_toolenv():
    # ensure the package is installed
    ensure_package_installed()

    # run a task using the toolenv
    try:
        task = Task(
            dataset=[
                Sample(
                    input="Please use the list_files tool to list the files in the current directory"
                )
            ],
            plan=[use_tools(list_files()), generate()],
            scorer=includes(),
            tool_environment="podman",
        )
        eval_result = await eval_async(
            task,
            model=get_model(
                "mockllm/model",
                custom_outputs=[
                    ModelOutput.for_tool_call(
                        model="mockllm/model",
                        tool_name="list_files",
                        tool_arguments={"dir": "."},
                    ),
                    ModelOutput.from_content(
                        model="mockllm/model",
                        content="just some text after that exhausting tool call",
                    ),
                ],
            ),
            log_level="debug",
        )

        assert len(eval_result) == 1
        assert (
            eval_result[0].samples[0].messages[-2].content
            == "Hello from the fake PodmanToolEnvironment!"
        )

    except Exception as ex:
        pytest.fail(f"Exception raised: {ex}")


def ensure_package_installed():
    try:
        import inspect_package  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-deps", "tests/test_package"]
        )
