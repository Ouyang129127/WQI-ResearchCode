import pandas as pd
import os

# Read water quality data
path = r"D:\WQIPaper\basicData\Extracted_WaterQuality.xlsx"
if not os.path.exists(path):
    print(f"NOT FOUND: {path}")
    exit()

df = pd.read_excel(path)
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head().to_string())
print(f"\nDescribe:")
print(df.describe().to_string())
print(f"\nDtypes:")
print(df.dtypes)
