import pandas as pd


def merge_trains(delay_df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    delay_df = delay_df.merge(
        station_df.rename(
            columns={
                "Nom_Gare": "Departure station",
                "Trigramme": "departure_station_trigrame",
                "Position géographique": "departure_station_geo",
                "Segment(s) DRG": "departure_station_segment",
                "Code commune": "departure_station_comunal_code",
                "Code_UIC": "departure_station_UIC_code",
                "Id_Gare": "departure_station_id",
            }
        )[
            [
                "Departure station",
                "departure_station_id",
                "departure_station_geo",
                "departure_station_trigrame",
                "departure_station_segment",
                "departure_station_comunal_code",
                "departure_station_UIC_code",
            ]
        ],
        on="Departure station",
        how="left",
    )
    delay_df = delay_df.merge(
        station_df.rename(
            columns={
                "Nom_Gare": "Arrival station",
                "Trigramme": "arrival_station_trigrame",
                "Position géographique": "arrival_station_geo",
                "Segment(s) DRG": "arrival_station_segment",
                "Code commune": "arrival_station_comunal_code",
                "Code_UIC": "arrival_station_UIC_code",
                "Id_Gare": "arrival_station_id",
            }
        )[
            [
                "Arrival station",
                "arrival_station_id",
                "arrival_station_geo",
                "arrival_station_trigrame",
                "arrival_station_segment",
                "arrival_station_comunal_code",
                "arrival_station_UIC_code",
            ]
        ],
        on="Arrival station",
        how="left",
    )
    return delay_df