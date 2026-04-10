import re
import pandas as pd
import numpy as np
from pathlib import Path

DK_FILE = "data/DKGroupWinners.txt"
FD_FILE = "data/FDGroupWinners.txt"
OUTPUT_FILE = "data/group_winner_odds.csv"


TEAM_ALIASES = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "Bosnia & Herzegovina": "Bosnia",
}


def normalize_team_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name.strip())
    return TEAM_ALIASES.get(name, name)


def american_to_implied_prob(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_prob_to_american(prob: float) -> int | None:
    if prob <= 0 or prob >= 1:
        return None
    if prob < 0.5:
        return round((100 / prob) - 100)
    return round(-(100 * prob) / (1 - prob))


def devig_probs(odds_by_team: dict[str, int]) -> dict[str, float]:
    implied = {team: american_to_implied_prob(odds) for team, odds in odds_by_team.items()}
    total = sum(implied.values())
    return {team: prob / total for team, prob in implied.items()}


def parse_dk_file(text: str) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}

    # Split on each group header
    pattern = re.compile(r"World Cup 2026\s+[–-]\s+Group ([A-L])\b", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        group_letter = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        odds_map: dict[str, int] = {}

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
    groups: dict[str, dict[str, int]] = {}

    pattern = re.compile(r"Group ([A-L]) Winner", re.MULTILINE)
    matches = list(pattern.finditer(text))

    i = 0
    while i < len(matches):
        match = matches[i]
        group_letter = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        odds_map: dict[str, int] = {}

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

        # Prefer the first populated occurrence for each group, since FD repeats headers
        if odds_map and group_letter not in groups:
            groups[group_letter] = odds_map

        i += 1

    return groups


def build_output(dk_groups: dict[str, dict[str, int]], fd_groups: dict[str, dict[str, int]]) -> pd.DataFrame:
    all_groups = sorted(set(dk_groups) | set(fd_groups))
    rows = []

    for group_letter in all_groups:
        dk = dk_groups.get(group_letter, {})
        fd = fd_groups.get(group_letter, {})

        dk_devig = devig_probs(dk) if dk else {}
        fd_devig = devig_probs(fd) if fd else {}

        all_teams = sorted(set(dk) | set(fd))

        for team in all_teams:
            probs = []
            odds_list = []

            if team in dk_devig:
                probs.append(dk_devig[team])
                odds_list.append(dk[team])

            if team in fd_devig:
                probs.append(fd_devig[team])
                odds_list.append(fd[team])

            if not probs:
                continue

            blended_prob = sum(probs) / len(probs)
            avg_american_odds = round(sum(odds_list) / len(odds_list))

            rows.append(
                {
                    "group": group_letter,
                    "team": team,
                    "draftkings_odds": dk.get(team),
                    "fanduel_odds": fd.get(team),
                    "draftkings_devig_prob": dk_devig.get(team),
                    "fanduel_devig_prob": fd_devig.get(team),
                    "books_used": len(probs),
                    "avg_american_odds": avg_american_odds,
                    "blended_devigged_probability": blended_prob,
                    "blended_devigged_percent": blended_prob * 100,
                    "blended_fair_american_odds": implied_prob_to_american(blended_prob),
                }
            )

    df = pd.DataFrame(rows)

    # Normalize within each group one more time, just in case of any parser edge cases
    df["blended_devigged_probability"] = (
        df.groupby("group")["blended_devigged_probability"]
        .transform(lambda s: s / s.sum())
    )
    df["blended_devigged_percent"] = df["blended_devigged_probability"] * 100
    df["blended_fair_american_odds"] = df["blended_devigged_probability"].apply(implied_prob_to_american)

    df = df.sort_values(["group", "blended_devigged_probability"], ascending=[True, False]).reset_index(drop=True)
    return df


def main() -> None:
    dk_text = Path(DK_FILE).read_text(encoding="utf-8")
    fd_text = Path(FD_FILE).read_text(encoding="utf-8")

    dk_groups = parse_dk_file(dk_text)
    fd_groups = parse_fd_file(fd_text)

    df = build_output(dk_groups, fd_groups)

    output_df = df[
        [
            "team",
            "avg_american_odds",
            "blended_devigged_probability",
            "blended_fair_american_odds",
        ]
    ].copy()

    output_df = output_df.rename(columns={
        "blended_devigged_probability": "blended_devigg_prob",
    })

    output_df["log_odds"] = np.log(
        output_df["blended_devigg_prob"] / (1 - output_df["blended_devigg_prob"])
    )

    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved CSV: {OUTPUT_FILE}")
    print(output_df.to_string(index=False))


if __name__ == "__main__":
    main()