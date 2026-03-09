import os
import pandas as pd

def load_bends_table(csv_path="outputs/bends_table.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find bends table: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def save_bends_with_clusters(df, out_path="outputs/bends_with_clusters.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
