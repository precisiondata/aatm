from typing import Dict, List
import copy


def format_prompt(prompt_template: List[Dict[str, str]], args: Dict[str, str]):
    prompt = copy.deepcopy(prompt_template)

    for message_id, message in enumerate(prompt):
        for field, text in message.items():
            if isinstance(text, str):
                for key, value in args.items():
                    prompt[message_id][field] = prompt[message_id][field].replace(
                        f"{{{key}}}", str(value)
                    )

    return prompt
