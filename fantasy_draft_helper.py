#!/usr/bin/env python3
"""
Fantasy Draft Helper

Targets: 12‑team PPR, roster: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), (K, DST optional)
Pick slot: #5 overall (snake by default)

Changes vs previous version:
----------------------------
- Replaced simple gap‑based tiering with a **statistical, SD‑based tiering system**:
  • For each position, players are sorted by projection.
  • Start tier 1 at the top.
  • For each player, compute whether adding them keeps the tier mean ± 2/3 SD constraint satisfied.
  • If yes, keep in tier; if not, start a new tier.
- This ensures positions like QB (where replacement value is tight) don’t collapse into a single giant tier.

Usage
-----
python fantasy_draft_helper.py build --out cheat_sheet.csv
python fantasy_draft_helper.py recommend --cheatsheet cheat_sheet.csv --team QB=0,RB=0,WR=0,TE=0 --drafted "Justin Jefferson,Christian McCaffrey"
"""

from __future__ import annotations
import argparse
import io
import sys
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

NFLVERSE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
PLAYER_WEEK_URL = NFLVERSE_BASE + "/player_stats/stats_player_week_{season}.csv"
PLAYERS_RELEASE = NFLVERSE_BASE + "/players/players.csv"

SUPPORTED_POS = {"QB", "RB", "WR", "TE"}

REPLACEMENT_RANK = {"QB": 12, "RB": 24, "WR": 24, "TE": 12}

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
    name_col = "full_name" if "full_name" in df.columns else ("display_name" if "display_name" in df.columns else None)
    pos_col = "position" if "position" in df.columns else ("pos" if "pos" in df.columns else None)
    id_col = None
    for candidate in ["gsis_id", "nfl_id", "pfr_id", "player_id"]:
        if candidate in df.columns:
            id_col = candidate
            break
    team_col = "team" if "team" in df.columns else ("recent_team" if "recent_team" in df.columns else None)
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
    df = weekly.copy()
    pid = None
    for c in ["player_id", "gsis_id", "nfl_id", "id"]:
        if c in df.columns:
            pid = c
            break
    ppr_col = None
    for c in ["fantasy_points_ppr", "fantasy_points", "ppr_points"]:
        if c in df.columns:
            ppr_col = c
            break
    grp = (
        df.groupby([pid, "season"], as_index=False)
          .agg(weeks_played=(ppr_col, "count"), ppr_points=(ppr_col, "sum"))
    )
    grp["ppg"] = grp["ppr_points"] / grp["weeks_played"].clip(lower=1)
    grp.rename(columns={pid: "player_id"}, inplace=True)
    return grp


def simple_projection(season_summ: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = season_summ.copy()
    last = df[df["season"] == cfg.seasons_end][["player_id", "ppg"]].rename(columns={"ppg": "ppg_last"})
    prev = df[df["season"] == (cfg.seasons_end - 1)][["player_id", "ppg"]].rename(columns={"ppg": "ppg_prev"})
    out = pd.merge(last, prev, on="player_id", how="left")
    out["ppg_prev"] = out["ppg_prev"].fillna(out["ppg_last"] * 0.85)
    out["ppg_proj"] = 0.7 * out["ppg_last"] + 0.3 * out["ppg_prev"]
    out["pts_proj"] = out["ppg_proj"] * cfg.expect_games
    return out


def attach_players(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(players, on="player_id", how="left")
    out = out[~out["pos"].isna()]
    out = out[out["pos"].isin(SUPPORTED_POS)]
    out.sort_values(["player", "pts_proj"], ascending=[True, False], inplace=True)
    out = out.drop_duplicates(subset=["player_id"])
    return out


def compute_vor(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vor_vals = []
    for pos, grp in df.groupby("pos"):
        grp = grp.sort_values("pts_proj", ascending=False).reset_index(drop=True)
        repl_rank = REPLACEMENT_RANK.get(pos, None)
        if repl_rank is None or len(grp) < repl_rank:
            baseline = grp["pts_proj"].quantile(0.25)
        else:
            baseline = grp.loc[repl_rank - 1, "pts_proj"]
        grp["vor"] = grp["pts_proj"] - baseline
        vor_vals.append(grp)
    return pd.concat(vor_vals, ignore_index=True)


# ----------------------------- Improved Tiering ----------------------------- #

def tier_sd_method(df: pd.DataFrame, sd_thresh: float = 2/3) -> pd.DataFrame:
    def tier_pos(grp: pd.DataFrame) -> pd.DataFrame:
        g = grp.sort_values("pts_proj", ascending=False).reset_index(drop=True)
        tiers = []
        current_tier = 1
        tier_players = []

        for idx, row in g.iterrows():
            if not tier_players:
                tier_players = [row["pts_proj"]]
                tiers.append(current_tier)
                continue
            tier_mean = np.mean(tier_players)
            tier_sd = np.std(tier_players) if len(tier_players) > 1 else 0.0
            # Check 2/3 SD criterion
            if abs(row["pts_proj"] - tier_mean) <= sd_thresh * (tier_sd if tier_sd > 0 else 1):
                # safe to include
                tier_players.append(row["pts_proj"])
                tiers.append(current_tier)
            else:
                # new tier
                current_tier += 1
                tier_players = [row["pts_proj"]]
                tiers.append(current_tier)
        g["tier"] = tiers
        return g
    return df.groupby("pos", group_keys=False).apply(tier_pos).reset_index(drop=True)


# ----------------------------- ADP Merge ----------------------------- #

def merge_adp(df: pd.DataFrame, adp_csv: Optional[str]) -> pd.DataFrame:
    if not adp_csv:
        df["adp_overall"] = np.nan
        return df
    adp = pd.read_csv(adp_csv)
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
    df = tier_sd_method(df)
    df = merge_adp(df, cfg.adp_csv)

    df["adp_boost"] = (-df["adp_overall"].fillna(200)) * 0.02
    df["score"] = df["vor"] + df["adp_boost"]

    df["pos_rank"] = df.sort_values("pts_proj", ascending=False).groupby("pos").cumcount() + 1
    board = df.sort_values(["score", "pts_proj"], ascending=[False, False]).reset_index(drop=True)

    cols = ["player", "team", "pos", "pos_rank", "tier", "pts_proj", "vor", "score", "ppg_proj", "adp_overall"]
    for c in cols:
        if c not in board.columns:
            board[c] = np.nan
    board = board[cols]
        # --- ADP Integration ---
    if "adp" not in out.columns:
        # TODO: replace with actual ADP import
        # Example: stub values or merge from a CSV
        out["adp"] = 100  # default fallback

    # Normalize ADP so lower = better
    out["adp_norm"] = 1 - (out["adp"] / out["adp"].max())

    # Blend into score
    out["score"] = (
        (out["ppg"] * 0.6) +     # projections
        (out["VOR"] * 0.25) +    # value over replacement
        (out["adp_norm"] * 0.15) # market anchor
    )
    board.to_csv(out_csv, index=False)
    return board


def roster_need_weights(team_counts: Dict[str, int]) -> Dict[str, float]:
    target = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    need = {}
    for pos in target:
        have = team_counts.get(pos, 0)
        remaining = max(target[pos] - have, 0)
        need[pos] = 1.0 + 0.4 * remaining
    return need

#-testing git-#
def recommend(cheatsheet_csv: str, team_counts_str: str, drafted_str: str = "", top_k: int = 10) -> pd.DataFrame:
    board = pd.read_csv(cheatsheet_csv)
    drafted = set([s.strip().lower() for s in drafted_str.split(",") if s.strip()])
    
    board = board[~board["player"].str.lower().isin(drafted)]

    def adjust_for_adp(df, pick, picks_per_round=12):
        next_pick = pick + picks_per_round
        
        # Risk flag: likely gone before your next pick
        df["risk_gone"] = (df["adp"] < next_pick).astype(int)
        
        # Adjusted score: slight bump for guys at risk
        df["score_adj"] = df["score"] + (0.2 * df["risk_gone"])
        
        return df.sort_values("score_adj", ascending=False)
    
    board = adjust_for_adp(board, pick=ns.pick)

    counts = {p: 0 for p in ["QB", "RB", "WR", "TE"]}
    if team_counts_str:
        for token in team_counts_str.split(","):
            if not token:
                continue
            pos, val = token.split("=")
            counts[pos.strip().upper()] = int(val)

    need_w = roster_need_weights(counts)
    board["need_adj_score"] = board.apply(lambda r: r["score"] * need_w.get(r["pos"], 1.0), axis=1)

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
    p = argparse.ArgumentParser(description="Fantasy Draft Helper (12‑team PPR)", formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build cheat sheet from nflverse data")
    b.add_argument("--out", required=True)
    b.add_argument("--since", type=int, default=2022)
    b.add_argument("--through", type=int, default=2024)
    b.add_argument("--expect_games", type=float, default=16.5)
    b.add_argument("--adp_csv", type=str, default=None)

    r = sub.add_parser("recommend", help="Recommend picks given roster + drafted list")
    r.add_argument("--cheatsheet", required=True)
    r.add_argument("--team", required=True)
    r.add_argument("--drafted", default="")
    r.add_argument("--topk", type=int, default=10)

    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    ns = parse_args(argv)
    if ns.cmd == "build":
        cfg = Config(seasons_start=ns.since, seasons_end=ns.through, expect_games=ns.expect_games, adp_csv=ns.adp_csv)
        board = build_cheatsheet(cfg, ns.out)
        print(f"Wrote {ns.out} with {len(board):,} players")
        return 0
    elif ns.cmd == "recommend":
        recs = recommend(ns.cheatsheet, ns.team, ns.drafted, ns.topk)
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(recs.to_string(index=False))
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
