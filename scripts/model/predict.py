import pandas as pd

from .config import SEASON_MAP


def predict_delay(
    departure: str,
    arrival: str,
    date: pd.Timestamp,
    model,
    route_stats: pd.DataFrame,
) -> float:
    """
    Predict average arrival delay (minutes) from two cities and a date.
    Used by the Streamlit dashboard — no hidden features required from the user.
    """
    row = route_stats[
        (route_stats["Departure station"] == departure)
        & (route_stats["Arrival station"] == arrival)
    ]
    fallback_label = None
    if row.empty:
        row = route_stats[route_stats["Departure station"] == departure]
        fallback_label = f"départ '{departure}'"
    if row.empty:
        row = route_stats
        fallback_label = "moyenne globale"
    if row.empty:
        raise ValueError(f"Route '{departure} → {arrival}' not found and no fallback available.")

    stats = row[["Average journey time", "Number of scheduled trains",
                 "Number of cancelled trains", "cancellation_rate"]].mean()
    service = row["Service"].mode().iloc[0] if not row.empty else "National"
    if fallback_label:
        import warnings
        warnings.warn(
            f"Route '{departure} → {arrival}' inconnue — estimation via {fallback_label}.",
            UserWarning,
            stacklevel=2,
        )

    month = date.month
    input_df = pd.DataFrame([{
        "Average journey time":       stats["Average journey time"],
        "Number of scheduled trains": stats["Number of scheduled trains"],
        "Number of cancelled trains": stats["Number of cancelled trains"],
        "cancellation_rate":          stats["cancellation_rate"],
        "year":                       date.year,
        "month":                      month,
        "day_of_week":                date.weekday(),
        "Departure station":          departure,
        "Arrival station":            arrival,
        "Service":                    service,
        "season":                     SEASON_MAP[month],
    }])
    # Support both plain pipeline and {"pipeline": ..., "model_name": ...} export
    pipeline = model.get("pipeline", model) if isinstance(model, dict) else model
    return float(pipeline.predict(input_df)[0])
