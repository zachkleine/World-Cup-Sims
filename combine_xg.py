from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input/xG"
OUTPUT_FILE = BASE_DIR / "combined_xg_wc_teams.csv"


WC_TEAMS = {
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "USA", "Paraguay", "Australia", "Turkey",
    "Germany", "Curacao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Sweden", "Tunisia",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Iraq", "Norway",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama",
}


NAME_MAP = {
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Bosnia": "Bosnia and Herzegovina",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Türkiye": "Turkey",
    "Turkey": "Turkey",
    "Czechia": "Czech Republic",
    "Czech Republic": "Czech Republic",
    "United States": "USA",
    "USA": "USA",
    "United States Men's": "USA",
    "United States Men's National Team": "USA",
    "Curaçao": "Curacao",
    "Curacao": "Curacao",
    "Congo DR": "DR Congo",
    "DR Congo": "DR Congo",
    "Cape Verde Islands": "Cape Verde",
}


def normalize_team_name(name: str) -> str:
    name = str(name).strip()
    return NAME_MAP.get(name, name)


def recency_weight(year: int) -> float:
    if year == 2026:
        return 1.00
    if year == 2025:
        return 0.85
    if year == 2024:
        return 0.65
    return 0.50


def get_competition_weight(competition: str, year: int) -> float:
    """
    Override competition weights in code so you don't need to edit every CSV.
    """
    competition = str(competition).strip().lower()

    if competition == "friendly":
        if year == 2026:
            return 0.65
        if year == 2025:
            return 0.60
        if year == 2024:
            return 0.45
        return 0.40

    if competition == "qualifier":
        return 1.00

    if competition == "league":
        return 0.90

    if competition == "tournament":
        return 1.15

    raise ValueError(f"Unknown competition type: {competition}")


def load_xg_file(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Rename from source format to model format
    df = df.rename(columns={
        "Team": "team",
        "MP": "matches",
        "xG": "xg_per_90",
        "xGA": "xga_per_90",
    })

    required = [
        "team",
        "xg_per_90",
        "xga_per_90",
        "matches",
        "year",
        "competition",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{filepath.name} missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Clean team names
    df["team"] = (
        df["team"]
        .astype(str)
        .str.replace(" National Team", "", regex=False)
        .str.strip()
        .map(normalize_team_name)
    )

    # Keep only what we need from the file
    keep_cols = [
        "team",
        "xg_per_90",
        "xga_per_90",
        "matches",
        "year",
        "competition",
    ]
    if "competition_weight" in df.columns:
        keep_cols.append("competition_weight")

    df = df[keep_cols].copy()

    # Convert types
    df["xg_per_90"] = pd.to_numeric(df["xg_per_90"], errors="coerce")
    df["xga_per_90"] = pd.to_numeric(df["xga_per_90"], errors="coerce")
    df["matches"] = pd.to_numeric(df["matches"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    bad = df[
        df[["xg_per_90", "xga_per_90", "matches", "year"]]
        .isna()
        .any(axis=1)
    ]
    if not bad.empty:
        raise ValueError(
            f"{filepath.name} has invalid values for teams: {bad['team'].tolist()}"
        )

    df["year"] = df["year"].astype(int)

    # Override CSV competition weights in code
    df["competition_weight"] = df.apply(
        lambda row: get_competition_weight(row["competition"], row["year"]),
        axis=1,
    )

    # Recency + weighting
    df["recency_weight"] = df["year"].apply(recency_weight)
    df["effective_weight"] = (
        df["matches"]
        * df["competition_weight"]
        * df["recency_weight"]
    )

    return df


def combine_xg() -> pd.DataFrame:
    files = sorted(INPUT_DIR.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {INPUT_DIR}")

    frames = [load_xg_file(f) for f in files]
    all_data = pd.concat(frames, ignore_index=True)

    combined = (
        all_data.groupby("team", as_index=False)
        .apply(
            lambda g: pd.Series({
                "weighted_xg_per_90": (
                    (g["xg_per_90"] * g["effective_weight"]).sum()
                    / g["effective_weight"].sum()
                ),
                "weighted_xga_per_90": (
                    (g["xga_per_90"] * g["effective_weight"]).sum()
                    / g["effective_weight"].sum()
                ),
                "weighted_xgd_per_90": (
                    (
                        (g["xg_per_90"] * g["effective_weight"]).sum()
                        / g["effective_weight"].sum()
                    )
                    -
                    (
                        (g["xga_per_90"] * g["effective_weight"]).sum()
                        / g["effective_weight"].sum()
                    )
                ),
                "total_matches": g["matches"].sum(),
                "total_effective_weight": g["effective_weight"].sum(),
                "sources_used": len(g),
                "unique_competitions": g["competition"].nunique(),
            })
        )
        .reset_index(drop=True)
    )

    combined["team"] = combined["team"].map(normalize_team_name)
    combined = combined[combined["team"].isin(WC_TEAMS)].copy()

    missing_wc_teams = sorted(WC_TEAMS - set(combined["team"]))
    if missing_wc_teams:
        print("Warning: these WC teams were not found in the xG inputs:")
        for team in missing_wc_teams:
            print(f" - {team}")

    combined = combined.sort_values(
        ["weighted_xgd_per_90", "weighted_xg_per_90"],
        ascending=False
    ).reset_index(drop=True)

    return combined


if __name__ == "__main__":
    combined = combine_xg()
    combined.to_csv(OUTPUT_FILE, index=False)

    print("\nCombined xG/xGA for qualified World Cup teams")
    print("-" * 110)
    print(combined.head(20).to_string(index=False))

    print("\nContribution by competition:")
    all_files = sorted(INPUT_DIR.glob("*.csv"))
    all_frames = [load_xg_file(f) for f in all_files]
    all_data = pd.concat(all_frames, ignore_index=True)
    print(
        all_data.groupby("competition")["effective_weight"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nContribution by competition and year:")
    print(
        all_data.groupby(["competition", "year"])["effective_weight"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print(f"\nWrote: {OUTPUT_FILE}")