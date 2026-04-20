from __future__ import annotations

from pathlib import Path
from typing import Iterable
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data/"

ELO_FILE = DATA_DIR / "elo_rating.csv"
GROUP_QUALIFY_FILE = DATA_DIR / "group_qualify_blended.csv"
GROUP_WINNER_FILE = DATA_DIR / "group_winner_odds.csv"
OUTRIGHT_FILE = DATA_DIR / "outright_winner_odds.csv"
SQUAD_VALUE_FILE = DATA_DIR / "SquadValue.csv"
COMBINED_XG_FILE = BASE_DIR / "combined_xg_wc_teams.csv"
OUTPUT_FILE = BASE_DIR / "team_strengths.csv"


def require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - series.mean()) / std


def normalize_team_name(name: str) -> str:
    name = str(name).strip()

    aliases = {
        "Bosnia & Herzegovina": "Bosnia and Herzegovina",
        "Bosnia Herzegovina": "Bosnia and Herzegovina",
        "Bosnia": "Bosnia and Herzegovina",
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
        "Democratic Republic of the Congo": "DR Congo",
        "DR Congo": "DR Congo",
        "Cape Verde Islands": "Cape Verde",
    }

    return aliases.get(name, name)


def load_elo(filepath: Path = ELO_FILE) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    require_columns(df, ["team", "strength", "one_year_change"], filepath.name)

    df = df[["team", "strength", "one_year_change"]].copy()
    df["team"] = df["team"].map(normalize_team_name)
    df["strength"] = pd.to_numeric(df["strength"], errors="coerce")
    df["one_year_change"] = pd.to_numeric(df["one_year_change"], errors="coerce")

    if df["strength"].isna().any():
        bad = df[df["strength"].isna()]["team"].tolist()
        raise ValueError(f"{filepath.name} has non-numeric strength values for: {bad}")

    if df["one_year_change"].isna().any():
        bad = df[df["one_year_change"].isna()]["team"].tolist()
        raise ValueError(f"{filepath.name} has non-numeric one_year_change values for: {bad}")

    return df.rename(columns={"strength": "elo_strength"})


def load_market_file(filepath: Path, log_odds_col_name: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    require_columns(df, ["team", "log_odds"], filepath.name)

    df = df[["team", "log_odds"]].copy()
    df["team"] = df["team"].map(normalize_team_name)
    df["log_odds"] = pd.to_numeric(df["log_odds"], errors="coerce")

    if df["log_odds"].isna().any():
        bad = df[df["log_odds"].isna()]["team"].tolist()
        raise ValueError(f"{filepath.name} has non-numeric log_odds for: {bad}")

    return df.rename(columns={"log_odds": log_odds_col_name})


def load_squad_value(filepath: Path = SQUAD_VALUE_FILE) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {
        "Team": "team",
        "Squad": "team",
        "Country": "team",
        "Market Value (€)": "squad_value_eur",
        "Market Value": "squad_value_eur",
        "squad_value_eur_m": "squad_value_eur",
    }
    df = df.rename(columns=rename_map)

    if "team" not in df.columns:
        raise ValueError(f"{filepath.name} must contain a team column")
    if "squad_value_eur" not in df.columns:
        raise ValueError(f"{filepath.name} must contain a squad value column")

    df = df[["team", "squad_value_eur"]].copy()
    df["team"] = df["team"].map(normalize_team_name)

    df["squad_value_eur"] = (
        df["squad_value_eur"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("m", "", regex=False)
        .str.replace("bn", "000", regex=False)
    )
    df["squad_value_eur"] = pd.to_numeric(df["squad_value_eur"], errors="coerce")

    if df["squad_value_eur"].isna().any():
        bad = df[df["squad_value_eur"].isna()]["team"].tolist()
        raise ValueError(f"{filepath.name} has non-numeric squad values for: {bad}")

    return df


def load_combined_xg(filepath: Path = COMBINED_XG_FILE) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    require_columns(
        df,
        [
            "team",
            "weighted_xg_per_90",
            "weighted_xga_per_90",
            "weighted_xgd_per_90",
            "total_matches",
            "total_effective_weight",
        ],
        filepath.name,
    )

    df = df[
        [
            "team",
            "weighted_xg_per_90",
            "weighted_xga_per_90",
            "weighted_xgd_per_90",
            "total_matches",
            "total_effective_weight",
        ]
    ].copy()
    df["team"] = df["team"].map(normalize_team_name)

    numeric_cols = [
        "weighted_xg_per_90",
        "weighted_xga_per_90",
        "weighted_xgd_per_90",
        "total_matches",
        "total_effective_weight",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad = df[df[numeric_cols].isna().any(axis=1)]
    if not bad.empty:
        raise ValueError(f"{filepath.name} has non-numeric xG data for: {bad['team'].tolist()}")

    return df


def confidence_factor(total_effective_weight: pd.Series, pivot: float = 14.0) -> pd.Series:
    """
    Shrinks xG influence for low-sample teams.
    """
    return (total_effective_weight / (total_effective_weight + pivot)).clip(0.25, 1.0)


def build_team_strengths(write_csv: bool = False) -> pd.DataFrame:
    elo = load_elo()
    qualify = load_market_file(GROUP_QUALIFY_FILE, "qual_log_odds")
    group_win = load_market_file(GROUP_WINNER_FILE, "group_log_odds")
    outright = load_market_file(OUTRIGHT_FILE, "outright_log_odds")
    squad = load_squad_value()
    xg = load_combined_xg()

    df = elo.merge(qualify, on="team", how="left")
    df = df.merge(group_win, on="team", how="left")
    df = df.merge(outright, on="team", how="left")
    df = df.merge(squad, on="team", how="left")
    df = df.merge(xg, on="team", how="left")

    required_after_merge = [
        "qual_log_odds",
        "group_log_odds",
        "outright_log_odds",
        "squad_value_eur",
        "weighted_xg_per_90",
        "weighted_xga_per_90",
        "weighted_xgd_per_90",
    ]

    missing_rows = df[df[required_after_merge].isna().any(axis=1)]
    if not missing_rows.empty:
        print("\nRows with missing merged inputs:")
        print(
            missing_rows[
                [
                    "team",
                    "qual_log_odds",
                    "group_log_odds",
                    "outright_log_odds",
                    "squad_value_eur",
                    "weighted_xg_per_90",
                    "weighted_xga_per_90",
                ]
            ].to_string(index=False)
        )
        raise ValueError(
            "Some teams are missing one or more inputs after merging: "
            f"{missing_rows['team'].tolist()}"
        )

    # Normalize base signals
    df["elo_z"] = zscore(df["elo_strength"])
    df["form_z"] = zscore(df["one_year_change"])
    df["qual_log_odds_z"] = zscore(df["qual_log_odds"])
    df["group_log_odds_z"] = zscore(df["group_log_odds"])
    df["outright_log_odds_z"] = zscore(df["outright_log_odds"])
    df["squad_value_z"] = zscore(df["squad_value_eur"])

    # Normalize xG signals
    df["xg_z"] = zscore(df["weighted_xg_per_90"])
    df["xga_z"] = zscore(df["weighted_xga_per_90"])
    df["xgd_z"] = zscore(df["weighted_xgd_per_90"])
    df["xg_confidence"] = confidence_factor(df["total_effective_weight"])

    # Overall strength: mostly Elo + markets, with xGD nudging it
    df["blended_strength_z"] = (
        0.30 * df["elo_z"]
        + 0.08 * df["qual_log_odds_z"]
        + 0.15 * df["group_log_odds_z"]
        + 0.22 * df["outright_log_odds_z"]
        + 0.10 * df["form_z"]
        + 0.10 * df["squad_value_z"]
        + 0.05 * (df["xgd_z"] * df["xg_confidence"])
    )

    # Compress spread so the sim doesn't get too top-heavy
    df["blended_strength_z"] *= 0.78

    # Convert back to an Elo-like scale
    elo_mean = df["elo_strength"].mean()
    blended_spread = 105.0
    df["overall_strength"] = elo_mean + blended_spread * df["blended_strength_z"]

    # Market/proxy-based attack & defense
    df["attack_market_proxy"] = 1.0 + (
        0.50 * df["group_log_odds_z"]
        + 0.20 * df["outright_log_odds_z"]
        + 0.15 * df["elo_z"]
        + 0.15 * df["squad_value_z"]
    ) * 0.10

    df["defense_market_proxy"] = 1.0 - (
        0.38 * df["qual_log_odds_z"]
        + 0.27 * df["elo_z"]
        + 0.10 * df["group_log_odds_z"]
        + 0.10 * df["squad_value_z"]
        + 0.15 * df["outright_log_odds_z"]
    ) * 0.08

    # Pure xG-derived attack & defense
    xg_mean = df["weighted_xg_per_90"].mean()
    xga_mean = df["weighted_xga_per_90"].mean()

    df["attack_xg_raw"] = (df["weighted_xg_per_90"] / xg_mean).clip(0.75, 1.30)
    df["defense_xg_raw"] = (df["weighted_xga_per_90"] / xga_mean).clip(0.75, 1.30)

    # Blend xG into attack/defense using confidence
    attack_xg_share = 0.28 * df["xg_confidence"]
    defense_xg_share = 0.28 * df["xg_confidence"]

    df["attack_rating"] = (
        (1 - attack_xg_share) * df["attack_market_proxy"]
        + attack_xg_share * df["attack_xg_raw"]
    )

    df["defense_rating"] = (
        (1 - defense_xg_share) * df["defense_market_proxy"]
        + defense_xg_share * df["defense_xg_raw"]
    )

    df["attack_rating"] = df["attack_rating"].clip(0.82, 1.22)
    df["defense_rating"] = df["defense_rating"].clip(0.82, 1.18)

    # Match-level modifiers
    df["form_rating"] = (df["form_z"] * 0.04).clip(-0.08, 0.08)
    df["squad_rating"] = (df["squad_value_z"] * 0.03).clip(-0.06, 0.06)
    df["tempo_factor"] = 1.0

    output_cols = [
        "team",
        "overall_strength",
        "attack_rating",
        "defense_rating",
        "form_rating",
        "tempo_factor",
        "squad_rating",
        "elo_strength",
        "one_year_change",
        "squad_value_eur",
        "qual_log_odds",
        "group_log_odds",
        "outright_log_odds",
        "weighted_xg_per_90",
        "weighted_xga_per_90",
        "weighted_xgd_per_90",
        "total_matches",
        "total_effective_weight",
        "xg_confidence",
        "elo_z",
        "form_z",
        "squad_value_z",
        "qual_log_odds_z",
        "group_log_odds_z",
        "outright_log_odds_z",
        "xg_z",
        "xga_z",
        "xgd_z",
        "blended_strength_z",
    ]

    df = df[output_cols].sort_values("overall_strength", ascending=False).reset_index(drop=True)

    if write_csv:
        df.to_csv(OUTPUT_FILE, index=False)

    return df


if __name__ == "__main__":
    strengths = build_team_strengths(write_csv=True)

    print("\nTop 20 blended team profiles")
    print("-" * 110)
    for i, row in enumerate(strengths.head(20).itertuples(index=False), start=1):
        print(
            f"{i:>2}. {row.team:<22} "
            f"overall={row.overall_strength:>7.2f}  "
            f"atk={row.attack_rating:>5.3f}  "
            f"def={row.defense_rating:>5.3f}  "
            f"xG={row.weighted_xg_per_90:>4.2f}  "
            f"xGA={row.weighted_xga_per_90:>4.2f}  "
            f"xGD={row.weighted_xgd_per_90:>5.2f}  "
            f"conf={row.xg_confidence:>4.2f}"
        )

    print(f"\nWrote: {OUTPUT_FILE}")