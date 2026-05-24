import pandas as pd

def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, sep=";")
    original_file = df.copy()
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df, original_file

def get_paths(csv_paths: str) -> list[str]:
    with open(csv_paths, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def load_meteo(csv_vent_path: str, csv_parameter_path) -> pd.DataFrame:
    csv_vent_paths = get_paths(csv_vent_path)
    csv_parameter_paths = get_paths(csv_parameter_path)
    vent_df = pd.concat([pd.read_csv(p, sep=";") for p in csv_vent_paths], ignore_index=True)
    parameter_df = pd.concat([pd.read_csv(p, sep=";") for p in csv_parameter_paths])
    meteo_df = pd.merge(vent_df, parameter_df, on=["NUM_POSTE", "AAAAMMJJ"])
    return meteo_df