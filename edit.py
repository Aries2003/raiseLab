import os
import pandas as pd

DATA_PATH = r"dataset\XSUM\validation-00000-of-00001 (1).parquet" 
OUTPUT_PATH = r"dataset\XSUM\validation-00000-of-00001 (1).parquet" 

def downsample_dataset():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Original dataset not found at {DATA_PATH}")
        return

    # 1. Load the original validation dataset
    df = pd.read_parquet(DATA_PATH)
    print(f"Original dataset size: {len(df)} rows.")

    # 2. Slice it down to just the first 500 rows
    df_500 = df.head(500).copy()

    # 3. Create destination folder if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # 4. Overwrite/save to your output destination
    df_500.to_parquet(OUTPUT_PATH, index=False)
    print(f"Successfully replaced! New file contains {len(df_500)} rows at: {OUTPUT_PATH}")

if __name__ == "__main__":
    downsample_dataset()