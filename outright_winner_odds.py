import os
import sys
from typing import Any, Dict, List

import pandas as pd
import requests

# =========================
# CONFIG
# =========================
API_KEY = os.environ.get("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "soccer_fifa_world_cup_winner"

BOOKMAKERS = [
    "draftkings",
    "pinnacle",
    "fanduel",
]

# Optional weights for consensus.
# You can tweak these later.
BOOK_WEIGHTS = {
    "DraftKings": 1.00,
    "Pinnacle": 1.35,
    "FanDuel": 1.00
}

if not API_KEY:
    raise ValueError("Please set the ODDS_API_KEY environment variable before running this script.")

# =========================
# API CALL
# =========================
def get_world_cup_outrights() -> tuple[list[dict[str, Any]], requests.structures.CaseInsensitiveDict]:
    url = f"{BASE_URL}/sports/{SPORT_KEY}/odds"

    params = {
        "apiKey": API_KEY,
        "markets": "outrights",
        "oddsFormat": "american",
        "bookmakers": ",".join(BOOKMAKERS),
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json(), response.headers

# =========================
# ODDS CONVERSION
# =========================
def american_to_implied_prob(odds: float | int | None) -> float | None:
    if odds is None:
        return None

    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

# =========================
# PARSE API RESPONSE
# =========================
def normalize_team_name(name: str) -> str:
    if not name:
        return name

    replacements = {
        "Türkiye": "Turkey",
        "USA": "United States",
        "U.S.A.": "United States",
        "Czechia": "Czech Republic",
    }

    name = name.strip()
    return replacements.get(name, name)


def parse_outrights(data):
    rows = []

    for event in data:
        event_id = event.get("id")
        sport_title = event.get("sport_title")
        commence_time = event.get("commence_time")

        for bookmaker in event.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key")
            bookmaker_title = bookmaker.get("title")
            last_update = bookmaker.get("last_update")

            for market in bookmaker.get("markets", []):
                if market.get("key") != "outrights":
                    continue

                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "sport_title": sport_title,
                        "commence_time": commence_time,
                        "bookmaker_key": bookmaker_key,
                        "bookmaker_title": bookmaker_title,
                        "market_key": market.get("key"),   # <-- add this back
                        "last_update": last_update,
                        "team": normalize_team_name(outcome.get("name")),
                        "odds_american": outcome.get("price"),
                    })

    return pd.DataFrame(rows)
# =========================
# CLEAN / LABEL EXCHANGE BACK-LAY MARKETS
# =========================
def normalize_bookmaker_name(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Collapse both Betfair Exchange keys into one display label.
    df["bookmaker_display"] = df["bookmaker_title"].replace(
        {
            "DraftKings": "DraftKings",
            "Pinnacle": "Pinnacle",
            "FanDuel": "Fanduel",
        }
    )

    return df

# =========================
# VIG REMOVAL
# =========================
def add_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["implied_prob"] = df["odds_american"].apply(american_to_implied_prob)
    return df

def remove_vig_by_bookmaker(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each bookmaker + market, normalize probabilities so they sum to 1.
    For outrights, this gives a simple fair-probability estimate within each book.
    """
    if df.empty:
        return df

    df = df.copy()

    group_cols = ["bookmaker_key", "market_key"]
    totals = (
        df.groupby(group_cols, dropna=False)["implied_prob"]
        .sum()
        .rename("book_total_implied_prob")
        .reset_index()
    )

    df = df.merge(totals, on=group_cols, how="left")
    df["fair_prob"] = df["implied_prob"] / df["book_total_implied_prob"]
    return df

# =========================
# CONSENSUS BUILD
# =========================
def build_consensus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces:
    - simple average fair probability across books
    - weighted average fair probability across books
    """
    if df.empty:
        return df

    consensus_source = (
        df[df["market_key"] == "outrights"]
        .copy()
        .dropna(subset=["team", "fair_prob", "bookmaker_display"])
    )

    if consensus_source.empty:
        return pd.DataFrame()

    consensus_source["weight"] = consensus_source["bookmaker_display"].map(BOOK_WEIGHTS).fillna(1.0)
    consensus_source["weighted_fair_prob_component"] = (
        consensus_source["fair_prob"] * consensus_source["weight"]
    )

    # Simple average
    simple = (
        consensus_source.groupby("team", as_index=False)
        .agg(
            books_used=("bookmaker_display", "nunique"),
            consensus_prob_simple=("fair_prob", "mean"),
            avg_american_odds=("odds_american", "mean"),
        )
    )

    # Weighted average
    weighted_num = (
        consensus_source.groupby("team", as_index=False)["weighted_fair_prob_component"]
        .sum()
        .rename(columns={"weighted_fair_prob_component": "weighted_sum"})
    )
    weighted_den = (
        consensus_source.groupby("team", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "weight_sum"})
    )

    weighted = weighted_num.merge(weighted_den, on="team", how="inner")
    weighted["consensus_prob_weighted"] = weighted["weighted_sum"] / weighted["weight_sum"]

    out = simple.merge(weighted[["team", "consensus_prob_weighted"]], on="team", how="left")

    # Normalize weighted consensus one more time so total sums to 1 across teams.
    total_weighted = out["consensus_prob_weighted"].sum()
    if total_weighted and total_weighted > 0:
        out["consensus_prob_weighted"] = out["consensus_prob_weighted"] / total_weighted

    total_simple = out["consensus_prob_simple"].sum()
    if total_simple and total_simple > 0:
        out["consensus_prob_simple"] = out["consensus_prob_simple"] / total_simple

    out["consensus_pct_weighted"] = out["consensus_prob_weighted"] * 100
    out["consensus_pct_simple"] = out["consensus_prob_simple"] * 100

    return out.sort_values("consensus_prob_weighted", ascending=False).reset_index(drop=True)

# =========================
# MAIN
# =========================
def main() -> None:
    try:
        data, headers = get_world_cup_outrights()
    except requests.HTTPError as exc:
        print("Request failed.")
        if exc.response is not None:
            print("Status code:", exc.response.status_code)
            print("Response:", exc.response.text)
        raise

    if not data:
        print("No data returned.")
        print("Possible reasons:")
        print("- This futures market is not currently available on your plan")
        print("- One or more requested bookmakers do not currently list this market")
        print("- The sport key is unavailable at the moment")
        sys.exit(0)

    raw_df = parse_outrights(data)

    if raw_df.empty:
        print("No outright rows found in the response.")
        print("Raw response:")
        print(data)
        sys.exit(0)

    raw_df = normalize_bookmaker_name(raw_df)
    raw_df = add_probabilities(raw_df)
    fair_df = remove_vig_by_bookmaker(raw_df)
    consensus_df = build_consensus(fair_df)

    output_df = consensus_df[["team", "avg_american_odds", "consensus_prob_weighted"]].copy()
    output_df = output_df.rename(columns={
        "team": "team",
        "avg_american_odds": "avg_american_odds",
        "consensus_prob_weighted": "implied_probability"
    })

    output_df.to_csv("data/outright_winner_odds.csv", index=False)
    print("\nSaved CSV: outright_winner_odds.csv")

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("\n=== RAW BOOKMAKER LINES ===\n")
    print(
        fair_df[
            [
                "team",
                "bookmaker_display",
                "bookmaker_key",
                "market_key",
                "odds_american",
                "implied_prob",
                "fair_prob",
                "last_update",
            ]
        ]
        .sort_values(["team", "bookmaker_display"])
        .to_string(index=False)
    )

    if consensus_df.empty:
        print("\nNo consensus table could be built.")
    else:
        print("\n=== CONSENSUS TRUE PROBABILITIES ===\n")
        print(
            consensus_df[
                [
                    "team",
                    "books_used",
                    "consensus_pct_weighted",
                    "consensus_pct_simple",
                    "avg_american_odds",
                ]
            ]
            .rename(
                columns={
                    "team": "Team",
                    "books_used": "BooksUsed",
                    "consensus_pct_weighted": "ConsensusPctWeighted",
                    "consensus_pct_simple": "ConsensusPctSimple",
                    "avg_american_odds": "AvgAmericanOdds",
                }
            )
            .to_string(index=False)
        )

    print("\n=== API USAGE ===")
    print("Requests remaining:", headers.get("x-requests-remaining"))
    print("Requests used:", headers.get("x-requests-used"))

if __name__ == "__main__":
    main()