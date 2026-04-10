import re
import math
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DK_FILE = DATA_DIR / "DKToQualify.txt"
FD_FILE = DATA_DIR / "FDToQualify.txt"
BET365_FILE = DATA_DIR / "Bet365ToQualify.csv"
OUTPUT_FILE = DATA_DIR / "group_qualify_blended.csv"

TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Bosnia & Herzegovina": "Bosnia",
    "Bosnia and Herzegovina": "Bosnia",
    "United States": "USA"
}


def normalize_team_name(name: str) -> str:
    name = re.sub(r"\s+", " ", str(name).strip())
    return TEAM_ALIASES.get(name, name)


def american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def prob_to_american(prob: float) -> int | None:
    if prob <= 0 or prob >= 1:
        return None
    if prob < 0.5:
        return round((100 / prob) - 100)
    return round(-(100 * prob) / (1 - prob))


def prob_to_log_odds(prob: float) -> float:
    clipped = min(max(prob, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def devig_yes_no(yes_odds: int, no_odds: int) -> float:
    yes_prob = american_to_implied_prob(yes_odds)
    no_prob = american_to_implied_prob(no_odds)
    total = yes_prob + no_prob
    return yes_prob / total if total > 0 else None


def parse_dk_file(text: str) -> dict[str, dict[str, int]]:
    groups = {}

    pattern = re.compile(r"World Cup 2026\s+[–-]\s+Group ([A-L])\b", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        group_letter = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        odds_map = {}

        j = 0
        while j < len(lines) - 1:
            team = lines[j]
            odds = lines[j + 1]

            if re.fullmatch(r"[+\-−]\d+", odds):
                odds = odds.replace("−", "-")
                odds_map[normalize_team_name(team)] = int(odds)
                j += 2
            else:
                j += 1

        groups[group_letter] = odds_map

    return groups


def parse_fd_file(text: str) -> dict[str, dict[str, int]]:
    groups = {}

    pattern = re.compile(r"Group ([A-L]) To Qualify", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        group_letter = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        odds_map = {}

        j = 0
        while j < len(lines) - 1:
            team = lines[j]
            odds = lines[j + 1]

            if re.fullmatch(r"[+\-−]\d+", odds):
                odds = odds.replace("−", "-")
                odds_map[normalize_team_name(team)] = int(odds)
                j += 2
            else:
                j += 1

        # FanDuel repeats the header, keep the first populated block
        if odds_map and group_letter not in groups:
            groups[group_letter] = odds_map

    return groups


def parse_bet365_csv(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    df["team"] = df["team"].apply(normalize_team_name)

    out = {}
    for _, row in df.iterrows():
        yes_odds = int(row["yes"])
        no_odds = int(row["no"])
        out[row["team"]] = devig_yes_no(yes_odds, no_odds)

    return out


def build_team_to_group_map(*group_dicts: dict[str, dict[str, int]]) -> dict[str, str]:
    team_to_group = {}
    for group_dict in group_dicts:
        for group_letter, odds_map in group_dict.items():
            for team in odds_map:
                team_to_group[team] = group_letter
    return team_to_group


def build_output(
    dk_groups: dict[str, dict[str, int]],
    fd_groups: dict[str, dict[str, int]],
    bet365_probs: dict[str, float],
) -> pd.DataFrame:
    rows = []

    team_to_group = build_team_to_group_map(dk_groups, fd_groups)
    all_teams = sorted(set(team_to_group) | set(bet365_probs))

    for team in all_teams:
        group_letter = team_to_group.get(team)

        dk_odds = None
        fd_odds = None
        dk_prob = None
        fd_prob = None
        b365_prob = bet365_probs.get(team)

        if group_letter:
            dk_group = dk_groups.get(group_letter, {})
            fd_group = fd_groups.get(group_letter, {})

            if team in dk_group:
                dk_odds = dk_group[team]
                dk_prob = american_to_implied_prob(dk_odds)

            if team in fd_group:
                fd_odds = fd_group[team]
                fd_prob = american_to_implied_prob(fd_odds)

        probs = [p for p in [dk_prob, fd_prob, b365_prob] if p is not None]
        odds_list = [o for o in [dk_odds, fd_odds] if o is not None]

        if not probs:
            continue

        blended_prob = sum(probs) / len(probs)
        avg_american_odds = round(sum(odds_list) / len(odds_list)) if odds_list else None

        rows.append({
            "group": group_letter,
            "team": team,
            "draftkings_odds": dk_odds,
            "fanduel_odds": fd_odds,
            "draftkings_implied_prob": dk_prob,
            "fanduel_implied_prob": fd_prob,
            "bet365_devig_prob": b365_prob,
            "books_used": len(probs),
            "avg_american_odds": avg_american_odds,
            "blended_devigged_probability": blended_prob,
            "blended_fair_american_odds": prob_to_american(blended_prob),
            "log_odds": prob_to_log_odds(blended_prob),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["group", "blended_devigged_probability"], ascending=[True, False]).reset_index(drop=True)
    return df


def main() -> None:
    dk_text = DK_FILE.read_text(encoding="utf-8")
    fd_text = FD_FILE.read_text(encoding="utf-8")

    dk_groups = parse_dk_file(dk_text)
    fd_groups = parse_fd_file(fd_text)
    bet365_probs = parse_bet365_csv(BET365_FILE)

    df = build_output(dk_groups, fd_groups, bet365_probs)

    output_df = df[
        [
            "team",
            "avg_american_odds",
            "blended_devigged_probability",
            "blended_fair_american_odds",
            "log_odds",
        ]
    ].copy()

    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved CSV: {OUTPUT_FILE}")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()