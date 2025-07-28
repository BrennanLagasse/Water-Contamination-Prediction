import pandas as pd

# Requires: pip install dbfread
from dbfread import DBF

# Load DBF into a DataFrame
table = DBF("soilmu_a_ca641.dbf copy")
df = pd.DataFrame(iter(table))

# Preview
print(df.head())