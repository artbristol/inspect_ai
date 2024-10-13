from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import exact
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import Tool, bash, tool


@tool
def ts() -> Tool:
    async def execute(x: int, y: int):
        """
        Add two numbers.

        Args:
            x: First number to add.
            y: Second number to add.

        Returns:
            The sum of the two numbers.
        """
        return x + y

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
        solver=[use_tools([bash(), ts()]), generate()],
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
                    tool_name="ts",
                    tool_arguments={"x": 5, "y": 19},
                ),
                ModelOutput.from_content("mockllm/model", content="wooza"),
            ],
        ),
    )
