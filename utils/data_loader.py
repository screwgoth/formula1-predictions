"""
Data loading utilities for F1 race prediction.
Uses FastF1 to fetch qualifying and race session data.
"""

import os
import warnings

import fastf1
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cache setup
# ---------------------------------------------------------------------------

def setup_cache(cache_dir: str = "data/cache") -> None:
    """Enable the FastF1 disk cache to avoid redundant API calls."""
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)


# ---------------------------------------------------------------------------
# Schedule helpers
# ---------------------------------------------------------------------------

def get_race_schedule(year: int) -> pd.DataFrame:
    """Return the event schedule for *year*, filtered to race weekends."""
    schedule = fastf1.get_event_schedule(year)
    valid_formats = {"conventional", "sprint_shootout", "sprint_qualifying", "sprint"}
    schedule = schedule[schedule["EventFormat"].isin(valid_formats)]
    return schedule


def get_round_number(year: int, event_name: str) -> int:
    """Look up the round number for a given event name."""
    schedule = get_race_schedule(year)
    match = schedule[schedule["EventName"].str.contains(event_name, case=False, na=False)]
    if match.empty:
        raise ValueError(
            f"'{event_name}' not found in {year} schedule. "
            f"Available: {schedule['EventName'].tolist()}"
        )
    return int(match.iloc[0]["RoundNumber"])


# ---------------------------------------------------------------------------
# Session loaders
# ---------------------------------------------------------------------------

def _load_session(year: int, event, session_type: str):
    """Load a FastF1 session with weather but without heavy telemetry."""
    session = fastf1.get_session(year, event, session_type)
    session.load(telemetry=False, weather=True, messages=False)
    return session


def load_race_results(year: int, event) -> pd.DataFrame | None:
    """Load race-session results and attach weather summary columns."""
    try:
        session = _load_session(year, event, "R")
        results = session.results.copy()
        results["Year"] = year
        results["EventName"] = (
            event if isinstance(event, str) else session.event["EventName"]
        )
        results["RoundNumber"] = int(session.event["RoundNumber"])

        # Attach aggregated weather
        weather = session.weather_data
        if weather is not None and not weather.empty:
            results["AirTemp"] = weather["AirTemp"].mean()
            results["TrackTemp"] = weather["TrackTemp"].mean()
            results["Humidity"] = weather["Humidity"].mean()
            results["WindSpeed"] = weather["WindSpeed"].mean()
            results["Rainfall"] = weather["Rainfall"].any()
        return results
    except Exception as exc:
        warnings.warn(f"Could not load Race for {year} {event}: {exc}")
        return None


def load_qualifying_results(year: int, event) -> pd.DataFrame | None:
    """Load qualifying results and best lap / sector times."""
    try:
        session = _load_session(year, event, "Q")
        results = session.results.copy()
        results["Year"] = year
        results["EventName"] = (
            event if isinstance(event, str) else session.event["EventName"]
        )
        results["RoundNumber"] = int(session.event["RoundNumber"])

        laps = session.laps
        if laps is not None and not laps.empty:
            best = (
                laps.groupby("DriverNumber")
                .agg(
                    QualifyingBestLap=("LapTime", "min"),
                    QualifyingBestS1=("Sector1Time", "min"),
                    QualifyingBestS2=("Sector2Time", "min"),
                    QualifyingBestS3=("Sector3Time", "min"),
                )
                .reset_index()
            )
            results = results.merge(best, on="DriverNumber", how="left")

        # Weather
        weather = session.weather_data
        if weather is not None and not weather.empty:
            results["AirTemp"] = weather["AirTemp"].mean()
            results["TrackTemp"] = weather["TrackTemp"].mean()
            results["Humidity"] = weather["Humidity"].mean()
            results["WindSpeed"] = weather["WindSpeed"].mean()
            results["Rainfall"] = weather["Rainfall"].any()

        return results
    except Exception as exc:
        warnings.warn(f"Could not load Qualifying for {year} {event}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Combined event loader
# ---------------------------------------------------------------------------

def collect_race_data(year: int, event) -> pd.DataFrame | None:
    """Merge qualifying info into race results for a single event."""
    race = load_race_results(year, event)
    if race is None:
        return None

    quali = load_qualifying_results(year, event)
    if quali is not None:
        quali_cols = [
            c for c in [
                "DriverNumber", "QualifyingBestLap",
                "QualifyingBestS1", "QualifyingBestS2", "QualifyingBestS3",
            ]
            if c in quali.columns
        ]
        if len(quali_cols) > 1:  # at least DriverNumber + one time col
            race = race.merge(quali[quali_cols], on="DriverNumber", how="left")
    return race


# ---------------------------------------------------------------------------
# Bulk collection helpers
# ---------------------------------------------------------------------------

def collect_historical_data(
    start_year: int, end_year: int, progress_callback=None
) -> pd.DataFrame:
    """Collect all race data from *start_year* to *end_year* inclusive."""
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        schedule = get_race_schedule(year)
        for _, row in schedule.iterrows():
            name = row["EventName"]
            if progress_callback:
                progress_callback(f"Loading {year} {name}…")
            data = collect_race_data(year, name)
            if data is not None:
                frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def collect_season_data(
    year: int, up_to_round: int, progress_callback=None
) -> pd.DataFrame:
    """Collect race data for *year*, rounds < *up_to_round*."""
    frames: list[pd.DataFrame] = []
    schedule = get_race_schedule(year)
    for _, row in schedule.iterrows():
        if row["RoundNumber"] >= up_to_round:
            break
        name = row["EventName"]
        if progress_callback:
            progress_callback(f"Loading {year} {name}…")
        data = collect_race_data(year, name)
        if data is not None:
            frames.append(data)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
