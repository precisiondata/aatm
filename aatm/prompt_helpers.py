"""
Provide utilities for formatting structured prompt templates.

This module contains helper functions for working with prompt templates
represented as lists of message dictionaries. Its main purpose is to replace
placeholder variables in string fields while preserving the original template
structure.
"""

from typing import Dict, List
import copy


def format_prompt(
    prompt_template: List[Dict[str, str]], args: Dict[str, str]
) -> List[Dict[str, str]]:
    """Format a prompt template by replacing placeholder variables.

    This function creates a deep copy of the input prompt template and replaces
    placeholders of the form ``{key}`` in each string field using the values
    provided in ``args``. The original template is not modified.

    Args:
        prompt_template: A list of message dictionaries representing the prompt
            template. Each dictionary typically contains fields such as roles
            and content.
        args: A mapping of placeholder names to replacement values. Each key is
            matched against placeholders in the template, and each value is
            converted to a string before substitution.

    Returns:
        A new prompt template with all matching placeholders replaced by their corresponding values.

    Notes:
        Only string fields are processed for placeholder replacement. Non-string values in the template are left unchanged.
    """

    prompt = copy.deepcopy(prompt_template)

    for message_id, message in enumerate(prompt):
        for field, text in message.items():
            if isinstance(text, str):
                for key, value in args.items():
                    prompt[message_id][field] = prompt[message_id][field].replace(
                        f"{{{key}}}", str(value)
                    )

    return prompt
