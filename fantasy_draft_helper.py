#!/usr/bin/env python3
"""
Fantasy Draft Helper — 1‑day mini‑project

Targets: 12‑team PPR, roster: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), (K, DST optional)
Pick slot: #5 overall (snake by default)

What it does
------------
- Pulls public weekly player stats from nflverse (2019‑2024) and aggregates to season level
- Estimates simple season projections (weighted PPG blend of last 2 seasons × expected games)
- Computes Value Over Replacement (VOR) using 12‑team replacement ranks
- Creates tiers per position using a gap-based method
- Produces a cheat sheet CSV and an interactive, roster‑aware recommender (CLI)
- Optionally merges an ADP CSV you provide (any source) to break ties

Usage
-----
python fantasy_draft_helper.py build --out cheat_sheet.csv
python fantasy_draft_helper.py recommend --cheatsheet cheat_sheet.csv --team QB=0,RB=0,WR=0,TE=0 --drafted "Justin Jefferson,Christian McCaffrey"

Optional flags:
  --adp_csv path/to/adp.csv  (columns: player, adp_overall; optional pos, team)
  --since 2021   (first season to include in history; default 2022)
  --expect_games 16.5  (expected games played for projections)

Notes
-----
• K and DST are ignored in projection math by default to keep things simple; you can still keep a tab open for late‑round K/DST.
• This is intentionally compact. If you want a model upgrade, add injury priors, target share projections, and schedule adjustments.

(tier-system is not working right)
- Important note - algorithm must factor in much more than j exp points... or avg points last season... need to normalize to position
    - for ex. I shouldn't see Lamar as a possible next best pick if in early round even w high PPG - hes a qb but the avg qb is also scoring a decent amount
    - value of replacement idea -> lamar isnt THAT much better than say a baker...
"""

from __future__ import annotations
import argparse
import io
import math
import sys
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
# nflverse weekly player offense CSV pattern (verified for recent seasons)
# e.g. https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_2024.csv
PLAYER_WEEK_URL = NFLVERSE_BASE + "/player_stats/stats_player_week_{season}.csv"
# Players master for positions/IDs (served as parquet/csv under 'players' release). We'll prefer CSV.
PLAYERS_RELEASE = NFLVERSE_BASE + "/players/players.csv"

SUPPORTED_POS = {"QB", "RB", "WR", "TE"}

REPLACEMENT_RANK = {  # 12‑team: QB12, RB24, WR24, TE12
    "QB": 12,
    "RB": 24,
    "WR": 24,
    "TE": 12,
}

@dataclass
class Config:
    seasons_start: int = 2022
    seasons_end: int = 2024
    expect_games: float = 16.5
    adp_csv: Optional[str] = None

# ------------------------------- Data Loading ------------------------------- #

def _http_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def load_players_master() -> pd.DataFrame:
    df = _http_csv(PLAYERS_RELEASE)
    # Standardize columns we need
    # Common fields: player_id (gsis_id or gsis_id-like), full_name, position, status, team
    cols = [c for c in df.columns]
    # Try to infer name and position columns
    name_col = "full_name" if "full_name" in cols else ("display_name" if "display_name" in cols else None)
    pos_col = "position" if "position" in cols else ("pos" if "pos" in cols else None)
    id_col = None
    for candidate in ["gsis_id", "nfl_id", "pfr_id", "player_id"]:
        if candidate in cols:
            id_col = candidate
            break
    team_col = "team" if "team" in cols else ("recent_team" if "recent_team" in cols else None)

    if not (name_col and pos_col and id_col):
        raise RuntimeError("Players master file schema changed; please inspect and update column mapping.")

    out = df[[id_col, name_col, pos_col] + ([team_col] if team_col else [])].copy()
    out.columns = ["player_id", "player", "pos"] + (["team"] if team_col else [])
    out["pos"] = out["pos"].str.upper()
    return out


def load_weekly_offense(seasons: List[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        url = PLAYER_WEEK_URL.format(season=s)
        try:
            df = _http_csv(url)
            df["season"] = s
            frames.append(df)
        except Exception as e:
            print(f"Warning: failed to load {url}: {e}")
    if not frames:
        raise RuntimeError("No weekly offense data loaded.")
    return pd.concat(frames, ignore_index=True)

# ----------------------------- Feature Engineering ----------------------------- #

def aggregate_to_season_ppg(weekly: pd.DataFrame) -> pd.DataFrame:
    # nflverse player weekly has columns: player_id (or gsis_id), fantasy_points_ppr, position, player_name sometimes
    # Normalize likely column names
    df = weekly.copy()
    # ID
    pid = None
    for c in ["player_id", "gsis_id", "nfl_id", "id"]:
        if c in df.columns:
            pid = c
            break
    if pid is None:
        raise RuntimeError("Could not find player id column in weekly data.")

    # name / pos might be missing here; we'll merge from players master
    # PPR column
    ppr_col = None
    for c in ["fantasy_points_ppr", "fantasy_points", "ppr_points"]:
        if c in df.columns:
            ppr_col = c
            break
    if ppr_col is None:
        raise RuntimeError("Weekly data missing PPR points column.")

    grp = (
        df.groupby([pid, "season"], as_index=False)
          .agg(weeks_played=(ppr_col, "count"),
               ppr_points=(ppr_col, "sum"))
    )
    grp["ppg"] = grp["ppr_points"] / grp["weeks_played"].clip(lower=1)
    grp.rename(columns={pid: "player_id"}, inplace=True)
    return grp


def simple_projection(season_summ: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    # Blend last two seasons PPG: 70% last season + 30% previous season
    df = season_summ.copy()
    last = df[df["season"] == cfg.seasons_end][["player_id", "ppg"]].rename(columns={"ppg": "ppg_last"})
    prev = df[df["season"] == (cfg.seasons_end - 1)][["player_id", "ppg"]].rename(columns={"ppg": "ppg_prev"})
    out = pd.merge(last, prev, on="player_id", how="left")
    out["ppg_prev"].fillna(out["ppg_last"] * 0.85, inplace=True)  # weak prior
    out["ppg_proj"] = 0.7 * out["ppg_last"] + 0.3 * out["ppg_prev"]
    out["pts_proj"] = out["ppg_proj"] * cfg.expect_games
    return out


def attach_players(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(players, on="player_id", how="left")
    out = out[~out["pos"].isna()]
    out = out[out["pos"].isin(SUPPORTED_POS)]
    # Deduplicate keeping highest projection
    out.sort_values(["player", "pts_proj"], ascending=[True, False], inplace=True)
    out = out.drop_duplicates(subset=["player_id"])  # keep best row per player
    return out


def compute_vor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vor_vals = []
    for pos, grp in df.groupby("pos"):
        grp = grp.sort_values("pts_proj", ascending=False).reset_index(drop=True)
        repl_rank = REPLACEMENT_RANK.get(pos, None)
        if repl_rank is None or len(grp) < repl_rank:
            baseline = grp["pts_proj"].quantile(0.25)  # fallback
        else:
            baseline = grp.loc[repl_rank - 1, "pts_proj"]
        grp["vor"] = grp["pts_proj"] - baseline
        vor_vals.append(grp)
    out = pd.concat(vor_vals, ignore_index=True)
    return out


def tier_by_gaps(df: pd.DataFrame, max_tiers: int = 8) -> pd.DataFrame:
    def tier_pos(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.sort_values("vor", ascending=False).reset_index(drop=True)
        diffs = g["vor"].diff(periods=-1).abs().fillna(0)
        threshold = np.nanpercentile(diffs, 70)  # big drop defines a tier break
        tier = 1
        tiers = []
        for i in range(len(g)):
            tiers.append(tier)
            if i < len(g) - 1 and diffs.iloc[i] >= threshold and tier < max_tiers:
                tier += 1
        g["tier"] = tiers
        return g
    out = (
        df.groupby("pos", group_keys=False)
          .apply(tier_pos)
          .reset_index(drop=True)
    )
    return out


def merge_adp(df: pd.DataFrame, adp_csv: Optional[str]) -> pd.DataFrame:
    if not adp_csv:
        df["adp_overall"] = np.nan
        return df
    adp = pd.read_csv(adp_csv)
    # Try flexible merge on player name
    for c in ["player", "name", "Player"]:
        if c in adp.columns:
            adp.rename(columns={c: "player"}, inplace=True)
            break
    if "adp_overall" not in adp.columns:
        raise RuntimeError("ADP CSV must have 'adp_overall' column.")
    merged = df.merge(adp[["player", "adp_overall"]], on="player", how="left")
    return merged


# ----------------------------- Public Interfaces ----------------------------- #

def build_cheatsheet(cfg: Config, out_csv: str) -> pd.DataFrame:
    seasons = list(range(cfg.seasons_start, cfg.seasons_end + 1))
    weekly = load_weekly_offense(seasons)
    season_ppg = aggregate_to_season_ppg(weekly)
    proj = simple_projection(season_ppg, cfg)
    players = load_players_master()
    df = attach_players(proj, players)
    df = compute_vor(df)
    df = tier_by_gaps(df)
    df = merge_adp(df, cfg.adp_csv)

    # Overall score: VOR with light ADP tiebreak (earlier ADP slightly boosts)
    df["adp_boost"] = (-df["adp_overall"].fillna(200)) * 0.02
    df["score"] = df["vor"] + df["adp_boost"]

    # Positional ranks
    df["pos_rank"] = df.sort_values("pts_proj", ascending=False).groupby("pos").cumcount() + 1
    # Overall board by score
    board = df.sort_values(["score", "pts_proj"], ascending=[False, False]).reset_index(drop=True)

    cols = [
        "player", "team", "pos", "pos_rank", "tier", "pts_proj", "vor", "score", "ppg_proj", "adp_overall"
    ]
    for c in cols:
        if c not in board.columns:
            board[c] = np.nan
    board = board[cols]
    board.to_csv(out_csv, index=False)
    return board


def roster_need_weights(team_counts: Dict[str, int]) -> Dict[str, float]:
    # Basic heuristic: encourage filling starting spots early; FLEX counts as RB/WR/TE equally
    target = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    need = {}
    for pos in target:
        have = team_counts.get(pos, 0)
        remaining = max(target[pos] - have, 0)
        need[pos] = 1.0 + 0.4 * remaining
    return need


def recommend(cheatsheet_csv: str, team_counts_str: str, drafted_str: str = "", top_k: int = 10) -> pd.DataFrame:
    board = pd.read_csv(cheatsheet_csv)
    drafted = set([s.strip().lower() for s in drafted_str.split(",") if s.strip()])
    board = board[~board["player"].str.lower().isin(drafted)]

    # Parse team counts string like "QB=0,RB=1,WR=1,TE=0"
    counts = {p: 0 for p in ["QB", "RB", "WR", "TE"]}
    if team_counts_str:
        for token in team_counts_str.split(","):
            if not token:
                continue
            pos, val = token.split("=")
            counts[pos.strip().upper()] = int(val)

    need_w = roster_need_weights(counts)

    # Adjust score by need
    board["need_adj_score"] = board.apply(lambda r: r["score"] * need_w.get(r["pos"], 1.0), axis=1)

    # Enforce simple roster sanity: don't take >3 QBs or >5 at a single pos early
    sane = []
    max_cap = {"QB": 3, "RB": 7, "WR": 7, "TE": 3}
    for pos, grp in board.groupby("pos"):
        grp = grp.head(max_cap.get(pos, 5))
        sane.append(grp)
    sane_board = pd.concat(sane).sort_values(["need_adj_score", "vor"], ascending=[False, False])

    recs = sane_board.head(top_k).copy()
    return recs[["player", "team", "pos", "pos_rank", "tier", "pts_proj", "vor", "need_adj_score"]]


# --------------------------------- CLI --------------------------------- #

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fantasy Draft Helper (12‑team PPR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python fantasy_draft_helper.py build --out cheat_sheet.csv --adp_csv my_adp.csv
              python fantasy_draft_helper.py recommend --cheatsheet cheat_sheet.csv --team QB=0,RB=1,WR=1,TE=0 --drafted "Justin Jefferson, CeeDee Lamb"
            """
        ),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build cheat sheet from nflverse data")
    b.add_argument("--out", required=True, help="Output CSV path for cheat sheet")
    b.add_argument("--since", type=int, default=2022, help="First season to include (default 2022)")
    b.add_argument("--through", type=int, default=2024, help="Last season to include (default 2024)")
    b.add_argument("--expect_games", type=float, default=16.5, help="Expected games played (default 16.5)")
    b.add_argument("--adp_csv", type=str, default=None, help="Optional ADP CSV to merge (player, adp_overall)")

    r = sub.add_parser("recommend", help="Recommend picks given roster + drafted list")
    r.add_argument("--cheatsheet", required=True, help="Cheat sheet CSV from the build step")
    r.add_argument("--team", required=True, help="Your current starters count, e.g. QB=0,RB=1,WR=1,TE=0")
    r.add_argument("--drafted", default="", help="Comma‑separated list of players already drafted")
    r.add_argument("--topk", type=int, default=10, help="How many recommendations to show")

    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    ns = parse_args(argv)
    if ns.cmd == "build":
        cfg = Config(
            seasons_start=ns.since,
            seasons_end=ns.through,
            expect_games=ns.expect_games,
            adp_csv=ns.adp_csv,
        )
        board = build_cheatsheet(cfg, ns.out)
        print(f"Wrote {ns.out} with {len(board):,} players")
        return 0
    elif ns.cmd == "recommend":
        recs = recommend(ns.cheatsheet, ns.team, ns.drafted, ns.topk)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(recs.to_string(index=False))
        return 0
    else:
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
