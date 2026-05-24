import re
import unicodedata

import pandas as pd

_STATION_NAME_MAP: dict[str, str] = {
    "BELLEGARDE (AIN)": "BELLEGARDE SUR VALSERINE",
    "DIJON VILLE": "DIJON",
    "LA ROCHELLE VILLE": "LA ROCHELLE",
    "LE CREUSOT MONTCEAU MONTCHANIN": "LE CREUSOT MONTCEAU LES MINES MONTCHANIN TGV",
    "LILLE": "LILLE FLANDRES",
    "MACON LOCHE": "MACON LOCHE TGV",
    "MARNE LA VALLEE": "MARNE LA VALLEE CHESSY",
    "MONTPELLIER": "MONTPELLIER SAINT ROCH",
    "MULHOUSE VILLE": "MULHOUSE",
    "NICE VILLE": "NICE",
    "PARIS LYON": "PARIS GARE DE LYON",
    "VALENCE ALIXAN TGV": "VALENCE TGV RHONE ALPES SUD",
}


def _normalize_station(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper().str.replace("-", " ", regex=False)
    s = s.apply(
        lambda x: unicodedata.normalize("NFD", x).encode("ascii", "ignore").decode("ascii")
    )
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def merge_trains(delay_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    station_norm = station_df.copy()
    station_norm["_key"] = _normalize_station(station_norm["Nom_Gare"])

    delay_df = delay_df.copy()
    delay_df["_dep_key"] = _normalize_station(delay_df["Departure station"]).map(
        lambda x: _STATION_NAME_MAP.get(x, x)
    )
    delay_df["_arr_key"] = _normalize_station(delay_df["Arrival station"]).map(
        lambda x: _STATION_NAME_MAP.get(x, x)
    )

    delay_df = delay_df.merge(
        station_norm.rename(
            columns={
                "_key": "_dep_key",
                "Trigramme": "departure_station_trigrame",
                "Position géographique": "departure_station_geo",
                "Segment(s) DRG": "departure_station_segment",
                "Code commune": "departure_station_comunal_code",
                "Code_UIC": "departure_station_UIC_code",
                "Id_Gare": "departure_station_id",
            }
        )[
            [
                "_dep_key",
                "departure_station_id",
                "departure_station_geo",
                "departure_station_trigrame",
                "departure_station_segment",
                "departure_station_comunal_code",
                "departure_station_UIC_code",
            ]
        ],
        on="_dep_key",
        how="left",
    ).drop(columns=["_dep_key"])

    delay_df = delay_df.merge(
        station_norm.rename(
            columns={
                "_key": "_arr_key",
                "Trigramme": "arrival_station_trigrame",
                "Position géographique": "arrival_station_geo",
                "Segment(s) DRG": "arrival_station_segment",
                "Code commune": "arrival_station_comunal_code",
                "Code_UIC": "arrival_station_UIC_code",
                "Id_Gare": "arrival_station_id",
            }
        )[
            [
                "_arr_key",
                "arrival_station_id",
                "arrival_station_geo",
                "arrival_station_trigrame",
                "arrival_station_segment",
                "arrival_station_comunal_code",
                "arrival_station_UIC_code",
            ]
        ],
        on="_arr_key",
        how="left",
    ).drop(columns=["_arr_key"])

    return delay_df
