import pandas as pd
from pathlib import Path
import jsonlines
from tqdm import tqdm
from copy import deepcopy
from datasets import Dataset

random_seed = 42

# Define paths
datasets_base_path = Path("datasets")
datasets_base_path.mkdir(exist_ok=True)
datasets_split_path = datasets_base_path / "splits"
datasets_split_path.mkdir(exist_ok=True)

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

# Define val and test data
for path in val_test_data_paths:
    df = pd.read_csv(path)
    val_test_dfs.append(df)

val_test_df: pd.DataFrame = pd.concat(val_test_dfs)
val_test_df = val_test_df.sample(frac=1, random_state=random_seed)
val_df = val_test_df.iloc[: int(0.5 * len(val_test_df))]
test_df = val_test_df.iloc[int(0.5 * len(val_test_df)) :]

# Save datasets in messages format for LLM instruction tuning
messages_template = {
    "messages": [
        {
            "role": "system",
            "content": "Convert natural language to a standard vocabulary code, which can be SNOMED, RxNorm or LOINC.",
        },
    ]
}


def format_message(msg: dict[str, str]) -> dict[str, str]:
    messages = deepcopy(messages_template)
    messages["messages"].append({"role": "user", "content": msg["prompt"]})
    messages["messages"].append({"role": "assistant", "content": msg["completion"]})
    return messages


# Load datasets in hf Datasets object
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

# Apply formatting to datasets
train_dataset = train_dataset.map(format_message)
val_dataset = val_dataset.map(format_message)
test_dataset = test_dataset.map(format_message)

# Save them to disk
train_dataset.save_to_disk(datasets_split_path / "train")
val_dataset.save_to_disk(datasets_split_path / "val")
test_dataset.save_to_disk(datasets_split_path / "test")

print(f"Train data shape: {train_df.shape}")
print(f"Val data shape: {val_df.shape}")
print(f"Test data shape: {test_df.shape}")
