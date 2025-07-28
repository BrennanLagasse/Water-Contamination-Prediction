import subprocess
import pandas as pd
from io import StringIO
import os

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Path to your MDB file
mdb_file = "Soil Column Names/soil_column_names.mdb"

# Step 1: List all table names
result = subprocess.run(
    ["mdb-tables", "-1", mdb_file],
    capture_output=True,
    text=True
)

table_names = result.stdout.strip().split('\n')
table_names = [name for name in table_names if not name.startswith("SYSTEM") and name]  # exclude system tables

# Step 2: Loop through and export/view each table
for table in table_names:
    print(f"\n--- Table: {table} ---")

    export = subprocess.run(
        ["mdb-export", mdb_file, table],
        capture_output=True,
        text=True
    )
    csv_data = export.stdout

    # Skip if empty
    if not csv_data.strip():
        print("(Empty table)")
        continue

    try:
        df = pd.read_csv(StringIO(csv_data))
        print(df.columns.tolist())
        print(f"Length of columns: {len(df.columns.tolist())}")
    except Exception as e:
        print(f"Could not load table {table}: {e}")