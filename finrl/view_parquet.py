# view_parquet.py

import argparse
import pandas as pd
from pathlib import Path

def main(n_files: int):
    data_path = Path("datasets")
    parquet_files = list(data_path.glob("*.parquet"))

    if not parquet_files:
        print("No .parquet files found in datasets/")
        return

    print(f"Found {len(parquet_files)} .parquet files. Displaying first {min(n_files, len(parquet_files))}:\n")

    for i, parquet_file in enumerate(sorted(parquet_files)):
        if i >= n_files:
            break
        print(f"==> File {i+1}: {parquet_file.name}")
        df = pd.read_parquet(parquet_file)
        print(df.head())
        print(df.dtypes)
        print(f"Rows: {len(df)}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display contents of .parquet files in datasets/")
    parser.add_argument("--n", type=int, default=1, help="Number of parquet files to display (default: 1)")
    args = parser.parse_args()

    main(args.n)
