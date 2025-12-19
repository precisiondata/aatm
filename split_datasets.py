import pandas as pd
from pathlib import Path
import jsonlines
from tqdm import tqdm
from copy import deepcopy

random_seed = 42

# Define paths
datasets_base_path = Path("datasets")
datasets_base_path.mkdir(exist_ok=True)
datasets_split_path = datasets_base_path / "splits"
datasets_split_path.mkdir(exist_ok=True)
datasets_split_msg_format_path = datasets_base_path / "splits-msg-format"
datasets_split_msg_format_path.mkdir(exist_ok=True)


train_data_paths = [
    "datasets/std_concept_to_std_vocab_code.csv",
    "datasets/std_concept_synonym_to_std_vocab_code.csv",
]

val_test_data_paths = [
    "datasets/non_std_concept_to_std_vocab_code.csv",
    "datasets/non_std_concept_synonym_to_std_vocab_code.csv",
]

train_dfs = []
val_test_dfs = []

# Define train data
for path in train_data_paths:
    df = pd.read_csv(path)
    train_dfs.append(df)

train_df: pd.DataFrame = pd.concat(train_dfs)
train_df = train_df.sample(frac=1, random_state=random_seed)
train_df.to_csv(datasets_split_path / "train.csv", index=False)

# Define val and test data
for path in val_test_data_paths:
    df = pd.read_csv(path)
    val_test_dfs.append(df)

val_test_df: pd.DataFrame = pd.concat(val_test_dfs)
val_test_df = val_test_df.sample(frac=1, random_state=random_seed)
val_df = val_test_df.iloc[: int(0.5 * len(val_test_df))]
val_df.to_csv(datasets_split_path / "val.csv", index=False)
test_df = val_test_df.iloc[int(0.5 * len(val_test_df)) :]
test_df.to_csv(datasets_split_path / "test.csv", index=False)

# Save datasets in messages format for LLM instruction tuning
messages_template = {
    "messages": [
        {
            "role": "system",
            "content": "Convert natural language to a standard vocabulary code, which can be SNOMED, RxNorm or LOINC.",
        },
    ]
}


def generate_messages(df: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    for _, row in df.iterrows():
        messages = deepcopy(messages_template)
        messages["messages"].append({"role": "user", "content": row["prompt"]})
        messages["messages"].append({"role": "assistant", "content": row["completion"]})
        yield messages


def save_messages(df: pd.DataFrame, path: Path):
    with jsonlines.open(path, mode="w") as writer:
        for msg in tqdm(
            generate_messages(df),
            desc="Adding msg format and writing to jsonl file",
            total=len(df),
        ):
            writer.write(msg)
    print(f"Saved {path}")


save_messages(train_df, datasets_split_msg_format_path / "train.jsonl")
save_messages(val_df, datasets_split_msg_format_path / "val.jsonl")
save_messages(test_df, datasets_split_msg_format_path / "test.jsonl")

print(f"Train data shape: {train_df.shape}")
print(f"Val data shape: {val_df.shape}")
print(f"Test data shape: {test_df.shape}")
