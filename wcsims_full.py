from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import csv
import json
import math
import random
from pathlib import Path
from collections import defaultdict, Counter

# =========================================================
# CONFIG
# =========================================================

RANDOM_SEED = 42
SIMULATION_COUNT = 10000

BASE_GOALS = 1.35
EXTRA_TIME_GOAL_MULTIPLIER = 0.33

random.seed(RANDOM_SEED)

# =========================================================
# DATA MODELS
# =========================================================

@dataclass
class Team:
    name: str
    group: str
    strength: float
    is_host: bool = False


@dataclass
class MatchResult:
    team_a: str
    team_b: str
    goals_a: int
    goals_b: int
    winner: Optional[str]
    stage: str
    went_to_extra_time: bool = False
    went_to_penalties: bool = False


@dataclass
class GroupTableRow:
    team: str
    group: str
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass
class TournamentResult:
    champion: str
    runner_up: str
    semifinalists: List[str]
    quarterfinalists: List[str]
    round_of_16: List[str]
    round_of_32: List[str]
    group_winners: List[str]
    advanced_third_place: List[str]
    games_played: Dict[str, int]


# =========================================================
# THIRD-PLACE LOOKUP
# =========================================================

THIRD_PLACE_WINNER_SLOTS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


def load_third_place_slot_map(path: Optional[Path] = None) -> Dict[Tuple[str, ...], List[str]]:
    """
    Loads a JSON mapping where:
      key   = "A-B-C-D-E-F-G-H"
      value = ["3H", "3G", "3B", "3C", "3A", "3F", "3D", "3E"]

    The value order corresponds to these winner slots:
      1A, 1B, 1D, 1E, 1G, 1I, 1K, 1L
    """
    if path is None:
        path = Path(__file__).with_name("third_place_slot_map.json")

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        tuple(key.split("-")): value
        for key, value in raw.items()
    }


THIRD_PLACE_SLOT_MAP = load_third_place_slot_map()


# =========================================================
# SAMPLE TEAMS
# Replace these later with real qualified teams / real draw
# =========================================================

def load_strengths_from_csv(filepath: str | Path) -> Dict[str, float]:
    strengths: Dict[str, float] = {}

    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            team = row["team"].strip()
            strength = float(row["strength"])
            strengths[team] = strength

    return strengths


def build_placeholder_teams(strengths: Dict[str, float]) -> List[Team]:
    group_map = {
        "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
        "B": ["Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland"],
        "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
        "D": ["United States", "Paraguay", "Australia", "Turkey"],
        "E": ["Germany", "Curacao", "Ivory Coast", "Ecuador"],
        "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
        "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
        "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
        "I": ["France", "Senegal", "Iraq", "Norway"],
        "J": ["Argentina", "Algeria", "Austria", "Jordan"],
        "K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
        "L": ["England", "Croatia", "Ghana", "Panama"],
    }

    hosts = {"USA", "Mexico", "Canada"}

    teams: List[Team] = []
    missing_teams: List[str] = []

    for group, team_names in group_map.items():
        for name in team_names:
            if name not in strengths:
                missing_teams.append(name)
                continue

            teams.append(
                Team(
                    name=name,
                    group=group,
                    strength=strengths[name],
                    is_host=name in hosts,
                )
            )

    if missing_teams:
        missing_str = ", ".join(sorted(missing_teams))
        raise ValueError(f"Missing strength values for: {missing_str}")

    return teams



# =========================================================
# UTILS
# =========================================================

def poisson_sample(lam: float) -> int:
    l = math.exp(-lam)
    k = 0
    p = 1.0
    while p > l:
        k += 1
        p *= random.random()
    return k - 1


def logistic(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def expected_goals(team_a: Team, team_b: Team, stage: str = "group") -> Tuple[float, float]:
    def elo_prob(a, b):
        return 1 / (1 + 10 ** ((b - a) / 400))

    p = elo_prob(team_a.strength, team_b.strength)

    base = 1.35
    scale = 2.0

    goal_diff = (p - 0.5) * scale

    lam_a = base + goal_diff
    lam_b = base - goal_diff

    return max(0.2, lam_a), max(0.2, lam_b)


def simulate_group_match(team_a: Team, team_b: Team) -> MatchResult:
    lam_a, lam_b = expected_goals(team_a, team_b, stage="group")
    goals_a = poisson_sample(lam_a)
    goals_b = poisson_sample(lam_b)

    winner = None
    if goals_a > goals_b:
        winner = team_a.name
    elif goals_b > goals_a:
        winner = team_b.name

    return MatchResult(
        team_a=team_a.name,
        team_b=team_b.name,
        goals_a=goals_a,
        goals_b=goals_b,
        winner=winner,
        stage="group",
    )


def simulate_knockout_match(team_a: Team, team_b: Team, stage: str) -> MatchResult:
    lam_a, lam_b = expected_goals(team_a, team_b, stage=stage)
    goals_a = poisson_sample(lam_a)
    goals_b = poisson_sample(lam_b)

    if goals_a != goals_b:
        winner = team_a.name if goals_a > goals_b else team_b.name
        return MatchResult(
            team_a=team_a.name,
            team_b=team_b.name,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            stage=stage,
        )

    et_lam_a = lam_a * EXTRA_TIME_GOAL_MULTIPLIER
    et_lam_b = lam_b * EXTRA_TIME_GOAL_MULTIPLIER
    et_a = poisson_sample(et_lam_a)
    et_b = poisson_sample(et_lam_b)

    goals_a += et_a
    goals_b += et_b

    if goals_a != goals_b:
        winner = team_a.name if goals_a > goals_b else team_b.name
        return MatchResult(
            team_a=team_a.name,
            team_b=team_b.name,
            goals_a=goals_a,
            goals_b=goals_b,
            winner=winner,
            stage=stage,
            went_to_extra_time=True,
        )

    penalty_win_prob_a = logistic((team_a.strength - team_b.strength) / 10.0)
    winner = team_a.name if random.random() < penalty_win_prob_a else team_b.name

    return MatchResult(
        team_a=team_a.name,
        team_b=team_b.name,
        goals_a=goals_a,
        goals_b=goals_b,
        winner=winner,
        stage=stage,
        went_to_extra_time=True,
        went_to_penalties=True,
    )


# =========================================================
# GROUP STAGE
# =========================================================

def group_fixtures(group_teams: List[Team]) -> List[Tuple[Team, Team]]:
    return [
        (group_teams[0], group_teams[1]),
        (group_teams[2], group_teams[3]),
        (group_teams[0], group_teams[2]),
        (group_teams[1], group_teams[3]),
        (group_teams[0], group_teams[3]),
        (group_teams[1], group_teams[2]),
    ]


def init_group_table(group_teams: List[Team]) -> Dict[str, GroupTableRow]:
    return {
        t.name: GroupTableRow(team=t.name, group=t.group)
        for t in group_teams
    }


def update_group_table(table: Dict[str, GroupTableRow], result: MatchResult) -> None:
    a = table[result.team_a]
    b = table[result.team_b]

    a.goals_for += result.goals_a
    a.goals_against += result.goals_b

    b.goals_for += result.goals_b
    b.goals_against += result.goals_a

    if result.goals_a > result.goals_b:
        a.points += 3
        a.wins += 1
        b.losses += 1
    elif result.goals_b > result.goals_a:
        b.points += 3
        b.wins += 1
        a.losses += 1
    else:
        a.points += 1
        b.points += 1
        a.draws += 1
        b.draws += 1


def sort_group_table(rows: List[GroupTableRow]) -> List[GroupTableRow]:
    return sorted(
        rows,
        key=lambda r: (
            r.points,
            r.goal_difference,
            r.goals_for,
            r.wins,
            random.random(),
        ),
        reverse=True,
    )


def simulate_group(group_teams: List[Team]) -> List[GroupTableRow]:
    table = init_group_table(group_teams)

    for team_a, team_b in group_fixtures(group_teams):
        result = simulate_group_match(team_a, team_b)
        update_group_table(table, result)

    rows = list(table.values())
    return sort_group_table(rows)


def simulate_all_groups(teams: List[Team]) -> Dict[str, List[GroupTableRow]]:
    by_group: Dict[str, List[Team]] = defaultdict(list)
    for team in teams:
        by_group[team.group].append(team)

    return {
        group: simulate_group(by_group[group])
        for group in sorted(by_group.keys())
    }


def select_best_third_place(group_results: Dict[str, List[GroupTableRow]]) -> List[GroupTableRow]:
    third_place_rows = [rows[2] for rows in group_results.values()]
    return sorted(
        third_place_rows,
        key=lambda r: (
            r.points,
            r.goal_difference,
            r.goals_for,
            r.wins,
            random.random(),
        ),
        reverse=True,
    )[:8]


# =========================================================
# KNOCKOUT BRACKET
# =========================================================

def team_lookup(teams: List[Team]) -> Dict[str, Team]:
    return {t.name: t for t in teams}


def build_slot_team_lookup(group_results: Dict[str, List[GroupTableRow]]) -> Dict[str, str]:
    """
    1A = winner of Group A
    2A = runner-up of Group A
    3A = third place in Group A
    """
    slot_to_team: Dict[str, str] = {}

    for group, rows in group_results.items():
        slot_to_team[f"1{group}"] = rows[0].team
        slot_to_team[f"2{group}"] = rows[1].team
        slot_to_team[f"3{group}"] = rows[2].team

    return slot_to_team


def build_round_of_32(
    group_results: Dict[str, List[GroupTableRow]],
    best_thirds: List[GroupTableRow],
) -> List[Tuple[str, str]]:
    """
    Returns Round of 32 pairings in official Match 73-88 order.

    Because the rest of the tournament uses adjacent winner propagation,
    this match order is what makes the later rounds line up correctly.
    """
    slot_to_team = build_slot_team_lookup(group_results)
    advancing_groups = tuple(sorted(row.group for row in best_thirds))

    if advancing_groups not in THIRD_PLACE_SLOT_MAP:
        raise ValueError(
            f"No third-place slot mapping found for advancing groups: {advancing_groups}"
        )

    mapped_third_slots = THIRD_PLACE_SLOT_MAP[advancing_groups]
    third_slot_for_winner = dict(zip(THIRD_PLACE_WINNER_SLOTS, mapped_third_slots))

    # Match 73 -> Match 88
    bracket_slot_pairs = [
        ("2A", "2B"),                                # 73
        ("1E", third_slot_for_winner["1E"]),         # 74
        ("1F", "2C"),                                # 75
        ("1C", "2F"),                                # 76
        ("1I", third_slot_for_winner["1I"]),         # 77
        ("2E", "2I"),                                # 78
        ("1A", third_slot_for_winner["1A"]),         # 79
        ("1L", third_slot_for_winner["1L"]),         # 80
        ("1D", third_slot_for_winner["1D"]),         # 81
        ("1G", third_slot_for_winner["1G"]),         # 82
        ("2K", "2L"),                                # 83
        ("1H", "2J"),                                # 84
        ("1B", third_slot_for_winner["1B"]),         # 85
        ("1J", "2H"),                                # 86
        ("1K", third_slot_for_winner["1K"]),         # 87
        ("2D", "2G"),                                # 88
    ]

    return [
        (slot_to_team[left_slot], slot_to_team[right_slot])
        for left_slot, right_slot in bracket_slot_pairs
    ]


def simulate_knockout_round(
    pairings: List[Tuple[str, str]],
    teams_by_name: Dict[str, Team],
    stage: str,
) -> Tuple[List[str], List[MatchResult]]:
    winners: List[str] = []
    results: List[MatchResult] = []

    for a_name, b_name in pairings:
        team_a = teams_by_name[a_name]
        team_b = teams_by_name[b_name]
        result = simulate_knockout_match(team_a, team_b, stage=stage)
        winners.append(result.winner)
        results.append(result)

    return winners, results


def next_round_pairings(qualified: List[str]) -> List[Tuple[str, str]]:
    return [
        (qualified[i], qualified[i + 1])
        for i in range(0, len(qualified), 2)
    ]


# =========================================================
# TOURNAMENT SIM
# =========================================================

def run_single_tournament(teams: List[Team]) -> TournamentResult:
    teams_by_name = team_lookup(teams)

    games_played = {t.name: 3 for t in teams}

    group_results = simulate_all_groups(teams)
    best_thirds = select_best_third_place(group_results)

    group_winners = [group_results[g][0].team for g in sorted(group_results.keys())]
    third_place_advancers = [row.team for row in best_thirds]

    r32_pairings = build_round_of_32(group_results, best_thirds)
    r32_participants = [team for match in r32_pairings for team in match]
    for team in r32_participants: 
        games_played[team] += 1

    r16_teams, _ = simulate_knockout_round(r32_pairings, teams_by_name, "R32")

    r16_pairings = next_round_pairings(r16_teams)
    for team in r16_teams:
        games_played[team] += 1

    qf_teams, _ = simulate_knockout_round(r16_pairings, teams_by_name, "R16")

    qf_pairings = next_round_pairings(qf_teams)
    
    for team in qf_teams:
        games_played[team] += 1

    sf_teams, _ = simulate_knockout_round(qf_pairings, teams_by_name, "QF")

    sf_pairings = next_round_pairings(sf_teams)
    for team in sf_teams:
        games_played[team] += 1

    finalists, sf_results = simulate_knockout_round(sf_pairings, teams_by_name, "SF")

    semifinal_losers: List[str] = []
    for result in sf_results:
        loser = result.team_b if result.winner == result.team_a else result.team_a
        semifinal_losers.append(loser)

    for team in semifinal_losers:
        games_played[team] += 1

    third_place_pairings = [(semifinal_losers[0], semifinal_losers[1])]
    simulate_knockout_round(third_place_pairings, teams_by_name, "ThirdPlace")

    for team in finalists:
        games_played[team] += 1
    final_pairings = [(finalists[0], finalists[1])]
    final_winner, final_results = simulate_knockout_round(final_pairings, teams_by_name, "Final")
    champion = final_winner[0]
    final_result = final_results[0]
    runner_up = final_result.team_b if final_result.winner == final_result.team_a else final_result.team_a

    return TournamentResult(
        champion=champion,
        runner_up=runner_up,
        semifinalists=sf_teams,
        quarterfinalists=qf_teams,
        round_of_16=r16_teams,
        round_of_32=r32_participants,
        group_winners=group_winners,
        advanced_third_place=third_place_advancers,
        games_played=games_played
    )


# =========================================================
# MONTE CARLO
# =========================================================

def run_monte_carlo(teams: List[Team], n: int = SIMULATION_COUNT) -> Dict[str, Dict[str, float]]:
    title_counts = Counter()
    final_counts = Counter()
    semifinal_counts = Counter()
    quarterfinal_counts = Counter()
    round16_counts = Counter()
    group_win_counts = Counter()
    best_third_counts = Counter()
    games_played_total = Counter()

    for _ in range(n):
        result = run_single_tournament(teams)

        title_counts[result.champion] += 1
        final_counts[result.champion] += 1
        final_counts[result.runner_up] += 1

        for team in result.semifinalists:
            semifinal_counts[team] += 1

        for team in result.quarterfinalists:
            quarterfinal_counts[team] += 1

        for team in result.round_of_16:
            round16_counts[team] += 1

        for team in result.group_winners:
            group_win_counts[team] += 1

        for team in result.advanced_third_place:
            best_third_counts[team] += 1

        for team, games in result.games_played.items():
            games_played_total[team] += games

    summary: Dict[str, Dict[str, float]] = {}
    for team in teams:
        summary[team.name] = {
            "win_world_cup_pct": 100 * title_counts[team.name] / n,
            "make_final_pct": 100 * final_counts[team.name] / n,
            "make_semis_pct": 100 * semifinal_counts[team.name] / n,
            "make_quarters_pct": 100 * quarterfinal_counts[team.name] / n,
            "make_round_of_16_pct": 100 * round16_counts[team.name] / n,
            "win_group_pct": 100 * group_win_counts[team.name] / n,
            "advance_as_best_third_pct": 100 * best_third_counts[team.name] / n,
            "avg_games_played": games_played_total[team.name] / n,
        }

    return summary


def print_top(summary: Dict[str, Dict[str, float]], metric: str, top_n: int = 15) -> None:
    ranked = sorted(summary.items(), key=lambda x: x[1][metric], reverse=True)

    print(f"\nTop {top_n} by {metric}:")
    print("-" * 60)
    for i, (team, stats) in enumerate(ranked[:top_n], start=1):
        print(f"{i:>2}. {team:<20} {stats[metric]:>6.2f}%")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    strengths_path = base_dir / "team_strengths.csv"
    strengths = load_strengths_from_csv(strengths_path)

    teams = build_placeholder_teams(strengths)
    summary = run_monte_carlo(teams, n=SIMULATION_COUNT)

    print_top(summary, "win_world_cup_pct", top_n=20)
    print_top(summary, "make_final_pct", top_n=20)
    print_top(summary, "make_semis_pct", top_n=20)
    print_top(summary, "avg_games_played", top_n=20)
