import pandas as pd

def merge_trains(delay_df: pd.DataFrame, station_df: pd.DataFrame) -> tuple[pd.DataFrame]:
    delay_df = delay_df.merge(
        station_df.rename(
            columns={
                "Nom_Gare": "Departure station",
                "Trigramme": "departure_station_trigrame",
                "Position géographique": "departure_station_geo",
                "Segment(s) DRG": "departure_station_segment",
                "Code commune": "",
                "Code_UIC": "",
                "Id_Gare": 
            }
        )[["Departure station", "departure_station_id", "departure_station_geo"]],
        on="Departure station",
        how="left",
    )
    delay_df = delay_df.merge(
        station_df.rename(
            columns={
                "station_name": "Arrival station",
                "station_id": "arrival_station_id",
                "geo_position": "arrival_station_geo",
            }
        )[["Arrival station", "arrival_station_id", "arrival_station_geo"]],
        on="Arrival station",
        how="left",
    )