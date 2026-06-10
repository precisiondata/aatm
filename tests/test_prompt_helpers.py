import pytest

import copy
from aatm.prompt_helpers import format_prompt


@pytest.mark.parametrize(
    "prompt_template, args, expected_result",
    [
        (
            [{"role": "user", "content": "Hello, {name}!"}],
            {"name": "Alice"},
            {
                "success": True,
                "expected_result": [{"role": "user", "content": "Hello, Alice!"}],
            },
        ),
        (
            [
                {"role": "user", "content": "Hello, {name}!"},
                {
                    "role": "assistant",
                    "content": "Hello, {name}! I'm {assistant_name}! How can I help you today?",
                },
                {"role": "user", "content": "I need help with my {problem_type}."},
            ],
            {"name": "Alice", "assistant_name": "Bob", "problem_type": "math"},
            {
                "success": True,
                "expected_result": [
                    {"role": "user", "content": "Hello, Alice!"},
                    {
                        "role": "assistant",
                        "content": "Hello, Alice! I'm Bob! How can I help you today?",
                    },
                    {
                        "role": "user",
                        "content": "I need help with my math.",
                    },
                ],
            },
        ),
        (
            [{"role": "user", "content": "Hello, {name}!"}],
            {"gender": "male"},
            {
                "success": True,
                "expected_result": [{"role": "user", "content": "Hello, {name}!"}],
            },
        ),
        (
            [{"role": "user", "content": 123456}],
            {"name": None},
            {
                "success": True,
                "expected_result": [{"role": "user", "content": 123456}],
            },
        ),
        (
            [{"role": "user", "content": [1, "asdf", "{name}"]}],
            {"name": None},
            {
                "success": True,
                "expected_result": [{"role": "user", "content": [1, "asdf", "{name}"]}],
            },
        ),
    ],
)
def test_format_prompt(prompt_template, args, expected_result):
    original_prompt_template = copy.deepcopy(prompt_template)
    result = format_prompt(prompt_template, args)

    assert original_prompt_template == prompt_template, (
        "Prompt template should not be modified"
    )
    assert result == expected_result["expected_result"], (
        "Prompt template not formatted correctly"
    )
