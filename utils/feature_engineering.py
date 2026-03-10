"""
Feature engineering pipeline for F1 race prediction.
All rolling / cumulative features use only past data to prevent leakage.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Circuit type lookup (expand as needed)
# ---------------------------------------------------------------------------
CIRCUIT_TYPES: dict[str, str] = {
    "Monaco": "street",
    "Singapore": "street",
    "Jeddah": "street",
    "Baku": "street",
    "Las Vegas": "street",
    "Miami": "hybrid",
    "Montreal": "hybrid",
    "Melbourne": "hybrid",
    "Zandvoort": "hybrid",
}


def _circuit_type(event_name: str) -> str:
    for key, ctype in CIRCUIT_TYPES.items():
        if key.lower() in event_name.lower():
            return ctype
    return "permanent"


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

def compute_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Grid, qualifying gap, season points, recent form, track history."""
    df = df.sort_values(["Year", "RoundNumber"]).copy()

    # Grid position
    df["grid_position"] = pd.to_numeric(df["GridPosition"], errors="coerce")

    # Qualifying delta to pole (seconds)
    if "QualifyingBestLap" in df.columns:
        df["quali_seconds"] = df["QualifyingBestLap"].dt.total_seconds()
        pole = df.groupby(["Year", "RoundNumber"])["quali_seconds"].transform("min")
        df["qualifying_time_delta"] = df["quali_seconds"] - pole
    else:
        df["qualifying_time_delta"] = np.nan

    # Finishing position (numeric target)
    df["finish_position"] = pd.to_numeric(df["Position"], errors="coerce")

    # Cumulative season points *before* this race
    df["driver_season_points"] = (
        df.groupby(["Year", "Abbreviation"])["Points"].cumsum() - df["Points"]
    )

    # Rolling-5 average finishing position (shifted so current race excluded)
    df["driver_recent_form"] = (
        df.groupby("Abbreviation")["finish_position"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    # Avg. finish at this circuit in prior visits
    df["driver_track_history"] = (
        df.groupby(["Abbreviation", "EventName"])["finish_position"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    return df


def compute_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constructor points, reliability, teammate gap."""
    df = df.copy()

    # Constructor cumulative points before this race
    team_pts = (
        df.groupby(["Year", "RoundNumber", "TeamName"])["Points"]
        .sum()
        .reset_index()
        .sort_values(["Year", "RoundNumber"])
    )
    team_pts["constructor_season_points"] = (
        team_pts.groupby(["Year", "TeamName"])["Points"].cumsum()
        - team_pts["Points"]
    )
    df = df.merge(
        team_pts[["Year", "RoundNumber", "TeamName", "constructor_season_points"]],
        on=["Year", "RoundNumber", "TeamName"],
        how="left",
    )

    # Reliability (rolling fraction of races finished)
    df["is_finished"] = df["Status"].apply(
        lambda s: 1 if s == "Finished" or (isinstance(s, str) and s.startswith("+")) else 0
    )
    df["team_reliability_rate"] = (
        df.groupby("TeamName")["is_finished"]
        .transform(lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    )

    # Teammate grid diff
    team_avg_grid = df.groupby(["Year", "RoundNumber", "TeamName"])["grid_position"].transform("mean")
    df["teammate_grid_diff"] = df["grid_position"] - team_avg_grid

    return df


def compute_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise weather columns."""
    df = df.copy()
    for col in ("AirTemp", "TrackTemp", "Humidity", "WindSpeed"):
        if col not in df.columns:
            df[col] = np.nan

    df["air_temp"] = pd.to_numeric(df["AirTemp"], errors="coerce")
    df["track_temp"] = pd.to_numeric(df["TrackTemp"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["Humidity"], errors="coerce")
    df["wind_speed"] = pd.to_numeric(df["WindSpeed"], errors="coerce")
    df["is_wet"] = df.get("Rainfall", pd.Series([False] * len(df))).astype(int)
    return df


def compute_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit type as a categorical feature."""
    df = df.copy()
    df["circuit_type"] = df["EventName"].apply(_circuit_type)
    return df


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "grid_position",
    "qualifying_time_delta",
    "driver_season_points",
    "driver_recent_form",
    "driver_track_history",
    "constructor_season_points",
    "team_reliability_rate",
    "teammate_grid_diff",
    "air_temp",
    "track_temp",
    "humidity",
    "wind_speed",
    "is_wet",
    "circuit_type",
]

META_COLS: list[str] = [
    "Year",
    "RoundNumber",
    "EventName",
    "Abbreviation",
    "TeamName",
]

TARGET_COL: str = "finish_position"


def build_feature_matrix(df: pd.DataFrame):
    """
    Run every feature pipeline and return (result_df, feature_cols, target_col).
    """
    df = compute_driver_features(df)
    df = compute_team_features(df)
    df = compute_weather_features(df)
    df = compute_circuit_features(df)

    avail_meta = [c for c in META_COLS if c in df.columns]
    avail_feat = [c for c in FEATURE_COLS if c in df.columns]

    result = df[avail_meta + avail_feat + [TARGET_COL]].copy()
    return result, avail_feat, TARGET_COL


def preprocess_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    fit: bool = True,
    scaler: StandardScaler | None = None,
    encoder: OrdinalEncoder | None = None,
):
    """
    Fill NaNs, encode categoricals, scale numerics.
    Returns (df, scaler, encoder, model_cols).
    """
    df = df.copy()

    numeric_cols = [c for c in feature_cols if c != "circuit_type"]
    cat_cols = [c for c in feature_cols if c == "circuit_type"]

    # Impute numerics with column median
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Encode categoricals
    if cat_cols:
        if fit:
            encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
            df[cat_cols] = encoder.fit_transform(df[cat_cols])
        else:
            df[cat_cols] = encoder.transform(df[cat_cols])

    model_cols = numeric_cols + cat_cols

    # Scale all model columns
    if fit:
        scaler = StandardScaler()
        df[model_cols] = scaler.fit_transform(df[model_cols])
    else:
        df[model_cols] = scaler.transform(df[model_cols])

    return df, scaler, encoder, model_cols
