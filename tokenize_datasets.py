"Dataset tokenization for training"

import copy
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer


def create_labels_for_all(input_ids, attention_mask):
    """
    Create labels for all tokens except padding (mask those with -100).
    """
    labels = []
    for i, mask in enumerate(attention_mask):
        if mask == 0:  # Padding token
            labels.append(-100)
        else:
            labels.append(input_ids[i])
    return labels


def create_masked_labels(messages, tokenizer, input_ids, attention_mask):
    """Create labels with input tokens masked (-100)"""
    labels = [-100] * len(input_ids)

    # Mask padding tokens in labels
    for i, mask in enumerate(attention_mask):
        if mask == 0:  # Padding token
            labels[i] = -100

    # Find assistant responses and unmask only those tokens
    for msg in messages:
        if msg["role"] == "assistant":
            assistant_content = msg["content"]

            # Find where this assistant response appears in the tokenized text
            assistant_tokens = tokenizer.encode(
                assistant_content, add_special_tokens=False
            )

            # Find the position of assistant response in input_ids
            decoded_assistant = [tokenizer.decode(item) for item in assistant_tokens]
            decoded_input = [tokenizer.decode(item) for item in input_ids]
            for i in range(len(input_ids) - len(assistant_tokens) + 1):
                if (
                    attention_mask[i] == 1
                    and decoded_input[i : i + len(assistant_tokens)]
                    == decoded_assistant
                ):
                    # Unmask the assistant response tokens
                    for j in range(i, min(i + len(assistant_tokens), len(input_ids))):
                        if attention_mask[j] == 1:  # Only unmask non-padding tokens
                            labels[j] = input_ids[j]
                    break

    return labels


def tokenize_conversations(
    batch: list[list[str, str]],
    tokenizer: AutoTokenizer = None,
    train_all: bool = False,  # train on every token or only assistant ones
    predictors: int = 0,  # n of predictor tokens
    plain=False,  # apply chat template or not
    front_pred=False,  # position of predictor tokens (front or not)
    reverse_pred=False,  # reverse prediction task to assistant msg -> user msg
    regular=False,  # use regular transformer or representation trainer
    max_length=512,
    debug=-1,
):
    input_ids_list = []
    labels_list = []
    attention_mask_list = []
    user_input_ids_list = []
    user_labels_list = []
    user_attention_mask_list = []
    assistant_input_ids_list = []
    assistant_labels_list = []
    assistant_attention_mask_list = []

    # Check presence of pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Pad token set to EOS token")

    # Tokenize conversations
    for msg_idx, msgs in enumerate(batch["messages"]):
        if debug == 0:
            print("Msg index", msg_idx)
            print("Msgs", msgs)

        # Format message
        formatted_msgs = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

        # Tokenize
        tokenized_msgs = tokenizer(
            formatted_msgs,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        input_ids = tokenized_msgs["input_ids"]
        attention_mask = tokenized_msgs["attention_mask"]

        # Create labels with proper masking
        if train_all:
            labels = create_labels_for_all(input_ids, attention_mask)
        else:
            labels = create_masked_labels(msgs, tokenizer, input_ids, attention_mask)

        # Append masked inputs and labels
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_mask_list.append(attention_mask)

        # Define prediction task
        ## user -> assistant or assistant -> user
        if reverse_pred:
            predictor_messages = copy.deepcopy(msgs[2:3])
        else:
            predictor_messages = copy.deepcopy(msgs[1:2])

        # Define and add predictor tokens
        to_add = predictors  # number of predictor tokens to add
        while to_add > 0:
            # Predictor tokens can be added at the beginning or in the end of the prompt
            if front_pred:
                predictor_messages[0]["content"] = (
                    f"<|predictor_{to_add}|>" + predictor_messages[0]["content"]
                )
            else:
                predictor_messages[0]["content"] += f"<|predictor_{to_add}|>"
            to_add -= 1

        # Prepare msgs that will be used for prediction
        if plain:
            formatted_chat_user = predictor_messages[0]["content"]
        else:
            formatted_chat_user = tokenizer.apply_chat_template(
                predictor_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        tokenized_user = tokenizer(
            formatted_chat_user,
            truncation=True,
            max_length=max_length,
            padding="max_length",  # Pad to max_length for consistent tensor shapes
            return_tensors=None,
        )
        user_input_ids_list.append(tokenized_user["input_ids"])
        user_labels_list.append([-100] * len(tokenized_user["input_ids"]))
        user_attention_mask_list.append(tokenized_user["attention_mask"])

        # Prepare target msgs
        if reverse_pred:
            target_messages = copy.deepcopy(msgs[1:2])
        else:
            target_messages = copy.deepcopy(msgs[2:3])

        if plain:
            formatted_target_messages = target_messages[0]["content"]
        else:
            formatted_target_messages = tokenizer.apply_chat_template(
                target_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        tokenized_assistant = tokenizer(
            formatted_target_messages,
            truncation=True,
            max_length=max_length,
            padding="max_length",  # Pad to max_length for consistent tensor shapes
            return_tensors=None,
        )
        assistant_input_ids_list.append(tokenized_assistant["input_ids"])
        assistant_labels_list.append([-100] * len(tokenized_assistant["input_ids"]))
        assistant_attention_mask_list.append(tokenized_assistant["attention_mask"])

    if regular:
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }
    else:
        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
            "input_ids_user": user_input_ids_list,
            "labels_user": user_labels_list,
            "attention_mask_user": user_attention_mask_list,
            "input_ids_assistant": assistant_input_ids_list,
            "labels_assistant": assistant_labels_list,
            "attention_mask_assistant": assistant_attention_mask_list,
        }


def load_and_prepare_dataset(
    dataset_path,
    tokenizer,
    predictors=0,
    regular=False,
    train_all=False,
    plain=False,
    front_pred=False,
    reverse_pred=False,
    max_length=512,
    debug=-1,
):
    dataset = load_from_disk(dataset_path)

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_conversations,
        batched=True,
        remove_columns=dataset.column_names,
        fn_kwargs={
            "tokenizer": tokenizer,
            "train_all": train_all,
            "predictors": predictors,
            "plain": plain,
            "front_pred": front_pred,
            "reverse_pred": reverse_pred,
            "regular": regular,
            "max_length": max_length,
            "debug": debug,
        },
    )

    return tokenized_dataset


if __name__ == "__main__":
    import argparse
    import yaml
    from dvclive import Live

    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str)
    args = parser.parse_args()

    with open(args.params, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    with Live(
        dir=Path(f'experiments/{params["exp_name"]}'),
    ) as live:
        tokenized_splits_base_path = Path("datasets/splits-tokenized")
        tokenized_splits_base_path.mkdir(exist_ok=True, parents=True)

        tokenizer = AutoTokenizer.from_pretrained(params["model_id"])
        for split in params["splits_to_use"]:
            tokenized_dataset = load_and_prepare_dataset(
                dataset_path=f"datasets/splits/{split}",
                tokenizer=tokenizer,
                predictors=params["predictors"],
                train_all=params["train_all"],
                plain=params["plain"],
                front_pred=params["front_pred"],
                reverse_pred=params["reverse_pred"],
                regular=params["regular"],
                max_length=params["max_length"],
                debug=params["debug"],
            )
            tokenized_dataset.save_to_disk(tokenized_splits_base_path / f"{split}")
