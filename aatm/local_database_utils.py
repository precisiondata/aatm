import pandas as pd
from pathlib import Path
import sqlite3
from rich.progress import track


def build_local_sqlite_vocab_database(vocab_dir: Path) -> None:
    con = sqlite3.connect("omop.db")
    list(vocab_dir.glob("*.csv"))[0].stem.lower()

    for file_path in track(list(vocab_dir.glob("*.csv")), description="Building vocabulary database..."):
        table_name = file_path.stem.lower()
        if table_name == "source_to_concept_map":
            df = pd.read_csv(file_path, sep=",", dtype=str)
        else:
            df = pd.read_csv(file_path, sep="\t", dtype=str)
        df.to_sql(table_name, con, index=False, if_exists="replace")

    con.commit()
    con.close()
