RANDOM_STATE = 42

TARGET = "Average delay of all trains at arrival"

ROUTE_STAT_FEATURES = [
    "Average journey time",
    "Number of scheduled trains",
    "Number of cancelled trains",
    "cancellation_rate",
]

NUMERIC_FEATURES = ROUTE_STAT_FEATURES + ["year", "month", "day_of_week"]
CATEGORICAL_FEATURES = ["Departure station", "Arrival station", "Service", "season"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SEASON_MAP = {
    1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn",
    11: "autumn", 12: "winter",
}
