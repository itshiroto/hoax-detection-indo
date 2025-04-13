import os
import pandas as pd

def main():
    dataset_path = os.getenv("DATASET_PATH", "hoax_1k.csv")
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {dataset_path}")
    print(f"Columns: {list(df.columns)}\n")

    for col in df.columns:
        max_len = df[col].astype(str).map(len).max()
        print(f"Column '{col}': max length = {max_len}")

if __name__ == "__main__":
    main()
