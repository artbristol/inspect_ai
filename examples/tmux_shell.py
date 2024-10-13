from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import bash


@task
def tmux_shell():
    return Task(
        dataset=[
            Sample(
                input="Just reply with Hello World",
                target="Hello World",
            )
        ],
        solver=[use_tools([bash()]), generate()],
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
                    tool_name="bash",
                    tool_arguments={"cmd": "echo foo"},
                ),
                ModelOutput.from_content("mockllm/model", content="wooza"),
            ],
        ),
    )
