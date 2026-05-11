import pandas as pd


def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, sep=";")
    original_file = df.copy()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, original_file
