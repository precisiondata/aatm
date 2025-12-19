import pandas as pd
from pathlib import Path
import sqlite3

con = sqlite3.connect("omop.db")
cur = con.cursor()
sus_vocab_path = Path("vocabularies")
list(sus_vocab_path.glob("*.csv"))[0].stem.lower()

for file_path in sus_vocab_path.glob("*.csv"):
    table_name = file_path.stem.lower()
    if table_name == "source_to_concept_map":
        df = pd.read_csv(file_path, sep=",", dtype=str)
    else:
        df = pd.read_csv(file_path, sep="\t", dtype=str)
    df.to_sql(table_name, con, index=False, if_exists="replace")

con.commit()
con.close()
