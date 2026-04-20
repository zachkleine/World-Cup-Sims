from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TEAM_SIM_FILE = BASE_DIR / "output" / "sim_results_20260419_204621.csv"   # change if needed
PLAYER_ODDS_FILE = BASE_DIR / "player_top_scorer_odds.csv"
OUTPUT_FILE = BASE_DIR / "output" / "player_projections.csv"


LISTED_SCORER_SHARE = 0.82


def american_to_implied_prob(odds: float) -> float:
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def load_team_sim(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    required = ["team", "avg_games_played", "avg_goals_scored"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filepath.name} missing required columns: {missing}")
    return df[required].copy()


def load_player_odds(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    required = ["player", "team", "american_odds", "minutes_share"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{filepath.name} missing required columns: {missing}")

    df = df.copy()
    df["american_odds"] = pd.to_numeric(df["american_odds"], errors="coerce")
    df["minutes_share"] = pd.to_numeric(df["minutes_share"], errors="coerce")
    df["penalty_share"] = pd.to_numeric(df.get("penalty_share", 0.0), errors="coerce").fillna(0.0)
    df["position"] = df.get("position", "UNK").fillna("UNK")

    bad = df[df[["american_odds", "minutes_share", "penalty_share"]].isna().any(axis=1)]
    if not bad.empty:
        raise ValueError(f"{filepath.name} has invalid numeric data for: {bad['player'].tolist()}")

    return df


def build_player_goal_projections(
    team_sim_file: Path = TEAM_SIM_FILE,
    player_odds_file: Path = PLAYER_ODDS_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    team_df = load_team_sim(team_sim_file)
    player_df = load_player_odds(player_odds_file)

    # Convert odds to implied probabilities
    player_df["raw_implied_prob"] = player_df["american_odds"].apply(american_to_implied_prob)

    # Normalize within team to get relative top-scorer weights
    player_df["normalized_top_scorer_weight"] = (
        player_df.groupby("team")["raw_implied_prob"]
        .transform(lambda s: s / s.sum())
    )

    # Minutes-adjusted scoring weight:
    # top-scorer markets already contain some minutes info implicitly,
    # but we still nudge using explicit minutes_share.
    player_df["minutes_adjusted_weight"] = (
        player_df["normalized_top_scorer_weight"] * player_df["minutes_share"]
    )

    player_df["goal_share"] = (
        player_df.groupby("team")["minutes_adjusted_weight"]
        .transform(lambda s: LISTED_SCORER_SHARE * s / s.sum())
    )

    # Small penalty-taker boost:
    # Add a light bonus for penalty role without letting it dominate.
    player_df["goal_share"] = player_df["goal_share"] * (1 + 0.12 * player_df["penalty_share"])

    # Renormalize after penalty adjustment
    player_df["goal_share"] = (
        player_df.groupby("team")["goal_share"]
        .transform(lambda s: LISTED_SCORER_SHARE * s / s.sum())
    )

    # Merge team opportunity
    df = player_df.merge(team_df, on="team", how="left")

    missing_team = df[df["avg_goals_scored"].isna()]
    if not missing_team.empty:
        raise ValueError(
            "Some players have teams missing from team sim output: "
            f"{missing_team[['player','team']].values.tolist()}"
        )

    # Core projections
    df["projected_goals"] = df["avg_goals_scored"] * df["goal_share"]
    df["goals_per_game"] = df["projected_goals"] / df["avg_games_played"]

    # Useful diagnostics
    df["team_goals_per_game"] = df["avg_goals_scored"] / df["avg_games_played"]

    output_cols = [
        "player",
        "team",
        "position",
        "american_odds",
        "minutes_share",
        "penalty_share",
        "raw_implied_prob",
        "normalized_top_scorer_weight",
        "goal_share",
        "avg_games_played",
        "avg_goals_scored",
        "team_goals_per_game",
        "projected_goals",
        "goals_per_game",
    ]

    df = df[output_cols].sort_values("projected_goals", ascending=False).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":
    projections = build_player_goal_projections()

    print("\nTop 30 projected tournament goal scorers")
    print("-" * 110)
    print(projections.head(30).to_string(index=False))

    print(f"\nWrote: {OUTPUT_FILE}")