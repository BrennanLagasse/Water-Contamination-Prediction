import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
mrds_df = pd.read_csv("MRDS US Mineral Chemicals v2.csv")

codes_per_row = mrds_df['CODE_LIST'].str.split()

all_codes = sorted(set(code for codes in codes_per_row for code in codes))
print(all_codes)
one_hot_df = pd.DataFrame(
    {code: codes_per_row.apply(lambda x: 1 if code in x else 0) for code in all_codes}
)
mrds_df = pd.concat([mrds_df, one_hot_df], axis=1)
mrds_df = mrds_df[["latitude", "longitude", "As-tot", "Cu-tot", "Pb-tot", "Hg-tot", "Cr-tot", "Fe-tot", "Mn-tot", "U-tot", "Mn-tot", "Fe-tot", "Zn-tot"]]
mrds_df.to_csv("MRDS v2 Cleaned.csv", index=False)
print(mrds_df.head())