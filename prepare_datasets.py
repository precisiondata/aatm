from pathlib import Path
import yaml
import pandas as pd
from pathlib import Path
import sqlite3
from tqdm import tqdm

datasets_base_path = Path("datasets")
datasets_base_path.mkdir(exist_ok=True)

con = sqlite3.connect("omop.db")
cur = con.cursor()

dfs = []

with open(Path("sql_commands.yaml"), "r") as f:
    sql_commands = yaml.safe_load(f)

for command_name in tqdm(sql_commands.keys()):
    sql_prompt = sql_commands[command_name]["sql"]
    sql_prompt_df = pd.read_sql(sql_prompt, con)
    sql_prompt_df["prompt"] = sql_prompt_df["concept_name"]
    sql_prompt_df["completion"] = (
        sql_prompt_df["vocabulary_id"] + " " + sql_prompt_df["concept_code"]
    )

    relevant_cols = [
        "concept_id",
        "prompt",
        "completion",
    ]

    sql_prompt_df[relevant_cols].to_csv(
        datasets_base_path / f"{command_name}.csv", index=False
    )
    dfs.append(sql_prompt_df[relevant_cols])
    print(command_name, sql_prompt_df[relevant_cols].shape)
