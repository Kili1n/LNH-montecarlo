#!/usr/bin/env python3
"""Mini serveur LNH Starligue : Dashboard historique + Monte Carlo (stdlib + pandas).
Lancer :  python server.py   puis ouvrir http://localhost:8000"""
from __future__ import annotations

import json, os, re, subprocess, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import simulation_montecarlo as sim

# ---- Configuration (en dur côté backend) -----------------------------------
DATA_DIR = "./data/data_csv"     # contient un sous-dossier par saison : match.csv, players.csv, players_boxscore.csv
OUTPUT_DIR = "./output"          # un sous-dossier par simulation lancée
PORT = int(os.environ.get("PORT", 8000))
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA, OUT = (HERE / DATA_DIR).resolve(), (HERE / OUTPUT_DIR).resolve()
JOBS: dict[str, dict] = {}
_cache: dict = {}

EXTRA_BOX_COLS = [
    "seconds_played", "goals_7m", "shots_7m", "saves_7m", "shots_faced_7m",
    "br", "tc", "pp", "2p", "bp", "dp", "a", "2min", "d",
]


def _stamp() -> float:
    fs = list(DATA.rglob("*.csv"))
    return max((f.stat().st_mtime for f in fs), default=0.0)


def _enrich_box_columns(box: pd.DataFrame) -> pd.DataFrame:
    """Garantit la présence des colonnes détaillées de players_boxscore.csv."""
    missing = [c for c in EXTRA_BOX_COLS if c not in box.columns]
    if missing:
        raw_frames = []
        for f in sim._find_csvs(DATA, "players_boxscore.csv"):
            try:
                raw_frames.append(pd.read_csv(f))
            except Exception:
                continue
        if raw_frames:
            raw = pd.concat(raw_frames, ignore_index=True)
            key_cols = [c for c in ["match_id", "jersey_num", "last_name"] if c in box.columns and c in raw.columns]
            keep = key_cols + [c for c in missing if c in raw.columns]
            if key_cols and len(keep) > len(key_cols):
                raw_sub = raw[keep].drop_duplicates(key_cols)
                box = box.merge(raw_sub, on=key_cols, how="left")
    for c in EXTRA_BOX_COLS:
        if c not in box.columns:
            box[c] = 0
        box[c] = pd.to_numeric(box[c], errors="coerce").fillna(0)
    return box


def load_data():
    """Charge matchs + feuilles de match une seule fois (rechargé si les CSV changent)."""
    st = _stamp()
    if _cache.get("stamp") == st and "m" in _cache:
        return _cache["m"], _cache["box"], _cache["seasons"]
    amap = sim.load_team_aliases(DATA)
    has_box = bool(sim._find_csvs(DATA, "players_boxscore.csv"))
    box = sim.load_player_tables(DATA, DATA, amap) if has_box else None
    m, seasons, _ = sim.load_all_matches(DATA, None, alias_map=amap)
    if box is not None and not m.empty:
        season_by_match = m.drop_duplicates("match_id").set_index("match_id")["season"]
        mapped = box["match_id"].map(season_by_match)
        box["season"] = mapped.fillna(box["season"].astype(str).str.replace("-", "/", regex=False))
        box = _enrich_box_columns(box)
    _cache.update(stamp=st, m=m, box=box, seasons=seasons)
    return m, box, seasons


def standings(df: pd.DataFrame) -> pd.DataFrame:
    p = df[df["played"]]
    h = pd.DataFrame({"t": p.home_team, "gf": p.home_ft_goals, "ga": p.away_ft_goals})
    a = pd.DataFrame({"t": p.away_team, "gf": p.away_ft_goals, "ga": p.home_ft_goals})
    x = pd.concat([h, a])
    x["pts"] = np.where(x.gf > x.ga, 2, np.where(x.gf == x.ga, 1, 0))
    g = x.groupby("t").agg(j=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"), pts=("pts", "sum"))
    g["gd"] = g.gf - g.ga
    g = g.sort_values(["pts", "gd", "gf"], ascending=False)
    g["rank"] = np.arange(1, len(g) + 1)
    return g.reset_index()


def halftime_stats(df: pd.DataFrame) -> list[dict]:
    """Calcule les statistiques mi-temps vs score final (remontadas, avances tenues, 2e période)."""
    p = df[df["played"]].copy()
    if p.empty or not {"home_ht_goals", "away_ht_goals"}.issubset(p.columns):
        return []
    p["h_ht"] = pd.to_numeric(p["home_ht_goals"], errors="coerce")
    p["a_ht"] = pd.to_numeric(p["away_ht_goals"], errors="coerce")
    p = p[p["h_ht"].notna() & p["a_ht"].notna() & ((p["h_ht"] + p["a_ht"]) > 0)]
    if p.empty:
        return []

    home = pd.DataFrame({
        "team": p["home_team"],
        "gf_ht": p["h_ht"], "ga_ht": p["a_ht"],
        "gf_ft": p["home_ft_goals"], "ga_ft": p["away_ft_goals"],
    })
    away = pd.DataFrame({
        "team": p["away_team"],
        "gf_ht": p["a_ht"], "ga_ht": p["h_ht"],
        "gf_ft": p["away_ft_goals"], "ga_ft": p["home_ft_goals"],
    })
    x = pd.concat([home, away], ignore_index=True)
    x["behind"] = (x["gf_ht"] < x["ga_ht"]).astype(int)
    x["cb_w"] = ((x["gf_ht"] < x["ga_ht"]) & (x["gf_ft"] > x["ga_ft"])).astype(int)
    x["cb_d"] = ((x["gf_ht"] < x["ga_ht"]) & (x["gf_ft"] == x["ga_ft"])).astype(int)
    x["ahead"] = (x["gf_ht"] > x["ga_ht"]).astype(int)
    x["kept"] = ((x["gf_ht"] > x["ga_ht"]) & (x["gf_ft"] > x["ga_ft"])).astype(int)
    x["blown"] = ((x["gf_ht"] > x["ga_ht"]) & (x["gf_ft"] <= x["ga_ft"])).astype(int)
    x["h2_gf"] = x["gf_ft"] - x["gf_ht"]
    x["h2_ga"] = x["ga_ft"] - x["ga_ht"]

    g = x.groupby("team").agg(
        j_ht=("gf_ft", "size"),
        behind_ht=("behind", "sum"),
        cb_w=("cb_w", "sum"),
        cb_d=("cb_d", "sum"),
        ahead_ht=("ahead", "sum"),
        kept=("kept", "sum"),
        blown=("blown", "sum"),
        h2_gf=("h2_gf", "sum"),
        h2_ga=("h2_ga", "sum"),
    ).reset_index()
    g["cb_pts"] = 2 * g["cb_w"] + g["cb_d"]
    g = g.sort_values(["cb_pts", "cb_w", "j_ht"], ascending=[False, False, True])
    return g.astype({
        "team": str, "j_ht": int, "behind_ht": int, "cb_w": int, "cb_d": int,
        "cb_pts": int, "ahead_ht": int, "kept": int, "blown": int,
    }).to_dict("records")


def top_players(box: pd.DataFrame | None, season: str | None) -> dict:
    empty = {"mvp": [], "scorers": [], "keepers": [], "seven_m": [], "defense": [], "discipline": []}
    if box is None:
        return empty
    b = box if season is None else box[box["season"] == season]
    if b.empty:
        return empty
    b = b.assign(name=(b["first_name"].fillna("") + " " + b["last_name"].fillna("")).str.strip())
    g = b.groupby(["name", "team"]).agg(
        games=("match_id", "nunique"),
        seconds=("seconds_played", "sum"),
        goals=("goals", "sum"),
        shots=("shots", "sum"),
        goals_7m=("goals_7m", "sum"),
        shots_7m=("shots_7m", "sum"),
        saves=("saves", "sum"),
        faced=("shots_faced", "sum"),
        saves_7m=("saves_7m", "sum"),
        faced_7m=("shots_faced_7m", "sum"),
        br=("br", "sum"),
        tc=("tc", "sum"),
        a=("a", "sum"),
        bp=("bp", "sum"),
        two_min=("2min", "sum"),
        lnh=("lnh_score", "sum"),
        gk=("is_gk", "max"),
    ).reset_index()

    g["minutes"] = (g["seconds"] / 60.0).round(1)
    g["lnh"] = g["lnh"].round(1)
    g["pct"] = np.where(g.shots > 0, 100 * g.goals / g.shots.clip(lower=1), 0).round(1)
    g["save_pct"] = np.where(g.faced > 0, 100 * g.saves / g.faced.clip(lower=1), 0).round(1)
    g["pct_7m"] = np.where(g.shots_7m > 0, 100 * g.goals_7m / g.shots_7m.clip(lower=1), 0).round(1)
    g["def_act"] = g["br"] + g["tc"]
    g["clean_idx"] = g["goals"] + g["a"] - g["bp"]

    cols = [
        "name", "team", "games", "minutes", "goals", "shots", "pct",
        "goals_7m", "shots_7m", "pct_7m", "saves", "faced", "save_pct",
        "saves_7m", "faced_7m", "br", "tc", "def_act", "a", "bp", "two_min", "lnh", "gk",
    ]
    rec = lambda d: d[cols].to_dict("records")
    min_faced = 60 if season else 150
    min_7m = 10 if season else 25
    return {
        "mvp": rec(g.sort_values("lnh", ascending=False).head(30)),
        "scorers": rec(g[~g.gk].sort_values("goals", ascending=False).head(30)),
        "keepers": rec(g[g.gk & (g.faced >= min_faced)].sort_values("save_pct", ascending=False).head(30)),
        "seven_m": rec(g[g.shots_7m >= min_7m].sort_values(["goals_7m", "pct_7m"], ascending=False).head(30)),
        "defense": rec(g[~g.gk].sort_values(["def_act", "br"], ascending=False).head(30)),
        "discipline": rec(g[~g.gk].sort_values(["a", "clean_idx"], ascending=False).head(30)),
    }


def build_history() -> dict:
    m, box, seasons = load_data()
    pl = m[m.played].copy()
    pl["tot"] = pl.home_ft_goals + pl.away_ft_goals
    per, ranks, teams_by, ht_by = [], {}, {}, {}
    for s in seasons:
        d = pl[pl.season == s]
        if d.empty:
            continue
        hw, dr = (d.home_ft_goals > d.away_ft_goals).mean(), (d.home_ft_goals == d.away_ft_goals).mean()
        per.append({"season": s, "matches": len(d), "goals": round(d.tot.mean(), 2), "home_win": round(100 * hw, 1),
                    "draw": round(100 * dr, 1), "home_adv": round((d.home_ft_goals - d.away_ft_goals).mean(), 2),
                    "max_total": int(d.tot.max())})
        st = standings(m[m.season == s])
        ranks[s] = st[["t", "rank", "pts", "gd", "j"]].rename(columns={"t": "team"}).to_dict("records")
        tm = st.assign(gf_m=(st.gf / st.j).round(2), ga_m=(st.ga / st.j).round(2))
        teams_by[s] = tm[["t", "gf_m", "ga_m", "pts", "j", "rank"]].rename(columns={"t": "team"}).to_dict("records")
        ht_by[s] = halftime_stats(m[m.season == s])
    allst = standings(m)
    allst = allst.assign(gf_m=(allst.gf / allst.j).round(2), ga_m=(allst.ga / allst.j).round(2))
    teams_by["all"] = allst[["t", "gf_m", "ga_m", "pts", "j", "rank"]].rename(columns={"t": "team"}).to_dict("records")
    ht_by["all"] = halftime_stats(m)
    players = {"all": top_players(box, None)}
    players.update({s: top_players(box, s) for s in seasons})
    big = pl.sort_values("tot", ascending=False).iloc[0]
    return {
        "kpis": {"seasons": len(per), "matches": int(len(pl)), "goals": round(pl.tot.mean(), 2),
                 "home_win": round(100 * (pl.home_ft_goals > pl.away_ft_goals).mean(), 1),
                 "home_adv": round((pl.home_ft_goals - pl.away_ft_goals).mean(), 2),
                 "record": {"total": int(big.tot), "score": f"{int(big.home_ft_goals)}-{int(big.away_ft_goals)}",
                            "match": f"{big.home_team} – {big.away_team}", "season": big.season}},
        "per_season": per, "ranks": ranks, "teams": teams_by, "players": players,
        "halftime": ht_by, "seasons": seasons,
    }


def meta() -> dict:
    m, _, seasons = load_data()
    info = {s: {"max_round": int(m[m.season == s]["round"].max()),
                "played_round": int(m[(m.season == s) & m.played]["round"].max()) if m[(m.season == s) & m.played].shape[0] else 0,
                "unplayed": int((~m[m.season == s].played).sum())} for s in seasons}
    return {"seasons": seasons, "info": info, "default": seasons[-1]}


# ---- Fiches détaillées Club & Joueur ----------------------------------------
def _filter_seasons_df(df: pd.DataFrame, seasons_raw: str) -> pd.DataFrame:
    if not seasons_raw or seasons_raw == "all":
        return df
    wanted = {s.strip() for s in seasons_raw.split(",") if s.strip()}
    return df[df["season"].isin(wanted)]


def api_club_detail(qs: dict) -> dict:
    team = qs.get("team", [""])[0].strip()
    seasons_raw = qs.get("seasons", ["all"])[0].strip()
    m, box, seasons = load_data()
    if not team:
        raise ValueError("Paramètre team manquant")

    m_all_team = m[m["played"] & ((m["home_team"] == team) | (m["away_team"] == team))].copy()
    m_per = _filter_seasons_df(m_all_team, seasons_raw)

    def _split_stats(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"j": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0, "gf_m": 0.0, "ga_m": 0.0, "win_pct": 0.0}
        is_h = sub["home_team"] == team
        gf = np.where(is_h, sub["home_ft_goals"], sub["away_ft_goals"])
        ga = np.where(is_h, sub["away_ft_goals"], sub["home_ft_goals"])
        w = int((gf > ga).sum())
        d = int((gf == ga).sum())
        l = int((gf < ga).sum())
        j = len(sub)
        return {
            "j": j, "w": w, "d": d, "l": l,
            "gf": int(gf.sum()), "ga": int(ga.sum()),
            "pts": 2 * w + d,
            "gf_m": round(float(gf.mean()), 2),
            "ga_m": round(float(ga.mean()), 2),
            "win_pct": round(100.0 * w / j, 1),
        }

    overall = _split_stats(m_per)
    home_st = _split_stats(m_per[m_per["home_team"] == team])
    away_st = _split_stats(m_per[m_per["away_team"] == team])

    # 5 derniers matchs sur la période
    m_sorted = m_per.sort_values(["date", "round"], ascending=[False, False])
    recent = []
    for r in m_sorted.head(5).itertuples():
        is_h = r.home_team == team
        gf = int(r.home_ft_goals if is_h else r.away_ft_goals)
        ga = int(r.away_ft_goals if is_h else r.home_ft_goals)
        res = "V" if gf > ga else ("N" if gf == ga else "D")
        recent.append({
            "date": str(r.date),
            "round": int(r.round),
            "season": str(r.season),
            "is_home": bool(is_h),
            "opponent": str(r.away_team if is_h else r.home_team),
            "score": f"{gf}–{ga}",
            "res": res,
        })

    # Salles les plus fréquentées sur la période
    venues = []
    if "venue" in m_per.columns and not m_per.empty:
        v_df = m_per[m_per["venue"].notna() & (m_per["venue"].astype(str).str.strip() != "")].copy()
        if not v_df.empty:
            is_h = v_df["home_team"] == team
            v_df["gf"] = np.where(is_h, v_df["home_ft_goals"], v_df["away_ft_goals"])
            v_df["ga"] = np.where(is_h, v_df["away_ft_goals"], v_df["home_ft_goals"])
            v_df["w"] = (v_df["gf"] > v_df["ga"]).astype(int)
            vg = v_df.groupby("venue").agg(j=("gf", "size"), w=("w", "sum"), gf=("gf", "mean"), ga=("ga", "mean")).reset_index()
            vg = vg.sort_values(["j", "w"], ascending=False).head(6)
            venues = [{
                "venue": str(r.venue),
                "j": int(r.j),
                "w": int(r.w),
                "win_pct": round(100.0 * r.w / r.j, 1),
                "gf_m": round(float(r.gf), 1),
                "ga_m": round(float(r.ga), 1),
            } for r in vg.itertuples()]

    # Effectif sur la période
    roster = []
    if box is not None and not box.empty:
        b_club = _filter_seasons_df(box[box["team"] == team], seasons_raw)
        if not b_club.empty:
            b_club = b_club.assign(name=(b_club["first_name"].fillna("") + " " + b_club["last_name"].fillna("")).str.strip())
            rg = b_club.groupby("name").agg(
                games=("match_id", "nunique"),
                seconds=("seconds_played", "sum"),
                goals=("goals", "sum"),
                shots=("shots", "sum"),
                saves=("saves", "sum"),
                faced=("shots_faced", "sum"),
                lnh=("lnh_score", "sum"),
                gk=("is_gk", "max"),
            ).reset_index().sort_values(["seconds", "lnh"], ascending=False)
            roster = [{
                "name": str(r.name),
                "games": int(r.games),
                "min_m": round(float(r.seconds) / 60.0 / max(int(r.games), 1), 1),
                "goals": int(r.goals),
                "shots": int(r.shots),
                "saves": int(r.saves),
                "faced": int(r.faced),
                "lnh": round(float(r.lnh), 1),
                "gk": bool(r.gk),
            } for r in rg.head(18).itertuples()]

    # Historique des classements du club sur toutes les saisons
    hist_ranks = []
    for s in seasons:
        st = standings(m[m["season"] == s])
        row = st[st["t"] == team]
        if not row.empty:
            r0 = row.iloc[0]
            hist_ranks.append({
                "season": s,
                "rank": int(r0["rank"]),
                "total_teams": int(len(st)),
                "pts": int(r0["pts"]),
                "j": int(r0["j"]),
                "gd": int(r0["gd"]),
            })

    return {
        "team": team,
        "overall": overall,
        "home": home_st,
        "away": away_st,
        "recent": recent,
        "venues": venues,
        "roster": roster,
        "history": hist_ranks,
    }


def api_player_detail(qs: dict) -> dict:
    name = qs.get("name", [""])[0].strip()
    seasons_raw = qs.get("seasons", ["all"])[0].strip()
    m, box, seasons = load_data()
    if not name or box is None or box.empty:
        raise ValueError("Joueur introuvable")

    b_all = box.assign(full_name=(box["first_name"].fillna("") + " " + box["last_name"].fillna("")).str.strip())
    b_per_all = _filter_seasons_df(b_all, seasons_raw)
    bp = b_per_all[b_per_all["full_name"] == name].copy()
    if bp.empty:
        bp = b_all[b_all["full_name"] == name].copy()
        b_per_all = b_all
    if bp.empty:
        raise ValueError(f"Aucune donnée pour {name}")

    is_gk = bool(bp["is_gk"].max())
    teams_list = list(dict.fromkeys(bp["team"].dropna().astype(str).tolist()))
    games = int(bp["match_id"].nunique())
    tot_min = float(bp["seconds_played"].sum()) / 60.0
    min_m = tot_min / max(games, 1)

    goals = int(bp["goals"].sum())
    shots = int(bp["shots"].sum())
    goals_7m = int(bp["goals_7m"].sum())
    shots_7m = int(bp["shots_7m"].sum())
    saves = int(bp["saves"].sum())
    faced = int(bp["shots_faced"].sum())
    saves_7m = int(bp["saves_7m"].sum())
    faced_7m = int(bp["shots_faced_7m"].sum())
    br = int(bp["br"].sum())
    tc = int(bp["tc"].sum())
    a = int(bp["a"].sum())
    bp_lost = int(bp["bp"].sum())
    two_min = int(bp["2min"].sum())
    lnh = float(bp["lnh_score"].sum())

    stats = {
        "games": games,
        "tot_min": round(tot_min, 1),
        "min_m": round(min_m, 1),
        "goals": goals,
        "shots": shots,
        "pct": round(100.0 * goals / max(shots, 1), 1) if shots > 0 else 0.0,
        "goals_7m": goals_7m,
        "shots_7m": shots_7m,
        "pct_7m": round(100.0 * goals_7m / max(shots_7m, 1), 1) if shots_7m > 0 else 0.0,
        "saves": saves,
        "faced": faced,
        "save_pct": round(100.0 * saves / max(faced, 1), 1) if faced > 0 else 0.0,
        "saves_7m": saves_7m,
        "faced_7m": faced_7m,
        "save_pct_7m": round(100.0 * saves_7m / max(faced_7m, 1), 1) if faced_7m > 0 else 0.0,
        "br": br,
        "tc": tc,
        "a": a,
        "bp": bp_lost,
        "two_min": two_min,
        "lnh": round(lnh, 1),
        "lnh_m": round(lnh / max(games, 1), 2),
    }

    # Radar normalisé (50 = moyenne de la ligue sur la période pour le même rôle)
    lg = b_per_all[(b_per_all["is_gk"] == is_gk) & (b_per_all["seconds_played"] >= 600)]
    if lg.empty:
        lg = b_per_all[b_per_all["is_gk"] == is_gk]
    lg_min = max(float(lg["seconds_played"].sum()) / 60.0, 1.0)
    p_min = max(tot_min, 1.0)

    def _norm(val: float, avg: float) -> int:
        if avg <= 0:
            return 50 if val <= 0 else 85
        return int(np.clip(round(50.0 * (val / avg)), 5, 100))

    if is_gk:
        lg_sv_pct = 100.0 * float(lg["saves"].sum()) / max(float(lg["shots_faced"].sum()), 1.0)
        lg_7m_pct = 100.0 * float(lg["saves_7m"].sum()) / max(float(lg["shots_faced_7m"].sum()), 1.0)
        lg_sv_60 = 60.0 * float(lg["saves"].sum()) / lg_min
        lg_lnh_60 = 60.0 * float(lg["lnh_score"].sum()) / lg_min
        lg_min_g = float(lg["seconds_played"].mean()) / 60.0
        radar = [
            {"axis": "% Arrêts", "score": _norm(stats["save_pct"], lg_sv_pct), "val": f"{stats['save_pct']} %", "avg": f"{lg_sv_pct:.1f} %"},
            {"axis": "% Arrêts 7m", "score": _norm(stats["save_pct_7m"], lg_7m_pct), "val": f"{stats['save_pct_7m']} %", "avg": f"{lg_7m_pct:.1f} %"},
            {"axis": "Arrêts / 60m", "score": _norm(60.0 * saves / p_min, lg_sv_60), "val": f"{60.0 * saves / p_min:.1f}", "avg": f"{lg_sv_60:.1f}"},
            {"axis": "Éval. LNH / 60m", "score": _norm(60.0 * lnh / p_min, lg_lnh_60), "val": f"{60.0 * lnh / p_min:.1f}", "avg": f"{lg_lnh_60:.1f}"},
            {"axis": "Temps / match", "score": _norm(min_m, lg_min_g), "val": f"{min_m:.1f} min", "avg": f"{lg_min_g:.1f} min"},
        ]
    else:
        lg_g_60 = 60.0 * float(lg["goals"].sum()) / lg_min
        lg_sh_pct = 100.0 * float(lg["goals"].sum()) / max(float(lg["shots"].sum()), 1.0)
        lg_a_60 = 60.0 * float(lg["a"].sum()) / lg_min
        lg_def_60 = 60.0 * float((lg["br"] + lg["tc"]).sum()) / lg_min
        lg_lnh_60 = 60.0 * float(lg["lnh_score"].sum()) / lg_min
        radar = [
            {"axis": "Buts / 60m", "score": _norm(60.0 * goals / p_min, lg_g_60), "val": f"{60.0 * goals / p_min:.1f}", "avg": f"{lg_g_60:.1f}"},
            {"axis": "Précision tir", "score": _norm(stats["pct"], lg_sh_pct), "val": f"{stats['pct']} %", "avg": f"{lg_sh_pct:.1f} %"},
            {"axis": "Passes D. / 60m", "score": _norm(60.0 * a / p_min, lg_a_60), "val": f"{60.0 * a / p_min:.1f}", "avg": f"{lg_a_60:.1f}"},
            {"axis": "Défense / 60m", "score": _norm(60.0 * (br + tc) / p_min, lg_def_60), "val": f"{60.0 * (br + tc) / p_min:.1f}", "avg": f"{lg_def_60:.1f}"},
            {"axis": "Éval. LNH / 60m", "score": _norm(60.0 * lnh / p_min, lg_lnh_60), "val": f"{60.0 * lnh / p_min:.1f}", "avg": f"{lg_lnh_60:.1f}"},
        ]

    # Historique par saison sur l'ensemble des données du joueur
    bp_full = b_all[b_all["full_name"] == name]
    sg = bp_full.groupby(["season", "team"]).agg(
        games=("match_id", "nunique"),
        seconds=("seconds_played", "sum"),
        goals=("goals", "sum"),
        shots=("shots", "sum"),
        saves=("saves", "sum"),
        faced=("shots_faced", "sum"),
        lnh=("lnh_score", "sum"),
    ).reset_index().sort_values("season")
    seasons_hist = [{
        "season": str(r.season),
        "team": str(r.team),
        "games": int(r.games),
        "min_m": round(float(r.seconds) / 60.0 / max(int(r.games), 1), 1),
        "goals": int(r.goals),
        "pct": round(100.0 * r.goals / max(int(r.shots), 1), 1) if r.shots > 0 else 0.0,
        "saves": int(r.saves),
        "save_pct": round(100.0 * r.saves / max(int(r.faced), 1), 1) if r.faced > 0 else 0.0,
        "lnh": round(float(r.lnh), 1),
    } for r in sg.itertuples()]

    # Top 5 matchs individuels sur la période
    meta_m = m[["match_id", "date", "round", "home_team", "away_team", "home_ft_goals", "away_ft_goals"]].drop_duplicates("match_id")
    bp_m = bp.merge(meta_m, on="match_id", how="left").sort_values(["lnh_score", "goals", "saves"], ascending=False)
    best_games = []
    for r in bp_m.head(5).itertuples():
        opp = r.away_team if r.team == r.home_team else r.home_team
        sc_txt = f"{int(r.home_ft_goals)}–{int(r.away_ft_goals)}" if pd.notna(r.home_ft_goals) else ""
        best_games.append({
            "date": str(r.date) if pd.notna(r.date) else "",
            "season": str(r.season),
            "team": str(r.team),
            "opponent": str(opp) if pd.notna(opp) else "",
            "score": sc_txt,
            "goals": int(r.goals),
            "shots": int(r.shots),
            "saves": int(r.saves),
            "faced": int(r.shots_faced),
            "lnh": round(float(r.lnh_score), 1),
        })

    return {
        "name": name,
        "teams": teams_list,
        "is_gk": is_gk,
        "stats": stats,
        "radar": radar,
        "seasons_history": seasons_hist,
        "best_games": best_games,
    }


# ---- Explorateur de données & Export CSV ------------------------------------
def _prepare_explorer_df(dataset: str) -> pd.DataFrame:
    m, box, _ = load_data()
    if dataset == "players":
        if box is None or box.empty:
            return pd.DataFrame()
        b = box.copy()
        if "date" not in b.columns or "round" not in b.columns:
            meta_m = m[["match_id", "date", "round", "home_team", "away_team"]].drop_duplicates("match_id")
            b = b.merge(meta_m, on="match_id", how="left")
        if "opponent" not in b.columns and {"home_team", "away_team"}.issubset(b.columns):
            b["opponent"] = np.where(b["team"] == b["home_team"], b["away_team"], b["home_team"])
        b["player"] = (b.get("first_name", pd.Series("", index=b.index)).fillna("") + " "
                       + b.get("last_name", pd.Series("", index=b.index)).fillna("")).str.strip()
        for col in ["goals", "shots", "saves", "shots_faced", "lnh_score", "seconds_played", "a", "br", "bp", "2min"]:
            if col not in b.columns:
                b[col] = 0
            b[col] = pd.to_numeric(b[col], errors="coerce").fillna(0)
        b["date"] = b.get("date", pd.Series("", index=b.index)).fillna("").astype(str)
        b["season"] = b.get("season", pd.Series("", index=b.index)).fillna("").astype(str)
        b["round"] = pd.to_numeric(b.get("round", 0), errors="coerce").fillna(0).astype(int)
        return b

    df = m.copy()
    df["total_goals"] = np.where(df["played"], df["home_ft_goals"] + df["away_ft_goals"], -1)
    df["score"] = np.where(
        df["played"],
        df["home_ft_goals"].fillna(0).astype(int).astype(str) + "–" + df["away_ft_goals"].fillna(0).astype(int).astype(str),
        "À venir",
    )
    has_ht = df["played"] & df["home_ht_goals"].notna() & df["away_ht_goals"].notna()
    df["ht_score"] = np.where(
        has_ht,
        df["home_ht_goals"].fillna(0).astype(int).astype(str) + "–" + df["away_ht_goals"].fillna(0).astype(int).astype(str),
        "–",
    )
    df["venue"] = df.get("venue", pd.Series("", index=df.index)).fillna("").astype(str)
    df["date"] = df["date"].fillna("").astype(str)
    df["round"] = pd.to_numeric(df["round"], errors="coerce").fillna(0).astype(int)
    return df


def _filter_explorer(qs: dict) -> tuple[str, pd.DataFrame, bool]:
    dataset = "players" if qs.get("dataset", ["matches"])[0] == "players" else "matches"
    full = qs.get("full", ["0"])[0] == "1"
    df = _prepare_explorer_df(dataset)
    if df.empty or full:
        return dataset, df, full

    seasons_raw = qs.get("seasons", [""])[0].strip()
    if seasons_raw and seasons_raw != "all":
        wanted = {s.strip() for s in seasons_raw.split(",") if s.strip()}
        df = df[df["season"].isin(wanted)]

    team = qs.get("team", [""])[0].strip()
    if team:
        if dataset == "players":
            df = df[df["team"] == team]
        else:
            df = df[(df["home_team"] == team) | (df["away_team"] == team)]

    q = qs.get("q", [""])[0].strip().lower()
    if q:
        cols_txt = ["player", "team", "opponent", "season"] if dataset == "players" else ["home_team", "away_team", "venue", "season"]
        mask = pd.Series(False, index=df.index)
        for c in cols_txt:
            if c in df.columns:
                mask |= df[c].fillna("").astype(str).str.lower().str.contains(re.escape(q), regex=True)
        df = df[mask]

    return dataset, df, full


def api_explorer(qs: dict) -> dict:
    dataset, df, _ = _filter_explorer(qs)
    if df.empty:
        return {"dataset": dataset, "total": 0, "page": 1, "pages": 1, "rows": []}

    sort_key = qs.get("sort", ["date"])[0]
    asc = qs.get("dir", ["desc"])[0] == "asc"

    if dataset == "players":
        sort_map = {
            "player": "player", "team": "team", "goals": "goals", "saves": "saves",
            "lnh_score": "lnh_score", "seconds_played": "seconds_played",
            "a": "a", "br": "br", "bp": "bp", "2min": "2min",
            "opponent": "opponent", "date": "date", "season": "season", "round": "round",
        }
        col = sort_map.get(sort_key, "lnh_score")
        df = df.sort_values([col, "date"], ascending=[asc, False], kind="mergesort")
    else:
        sort_map = {
            "date": "date", "home_team": "home_team", "total_goals": "total_goals",
            "away_team": "away_team", "ht_score": "ht_score", "round": "round",
            "season": "season", "venue": "venue",
        }
        col = sort_map.get(sort_key, "date")
        df = df.sort_values([col, "round"], ascending=[asc, False], kind="mergesort")

    page_size = int(np.clip(int(qs.get("page_size", [50])[0]), 10, 200))
    total = len(df)
    pages = max(1, (total + page_size - 1) // page_size)
    page = int(np.clip(int(qs.get("page", [1])[0]), 1, pages))
    sub = df.iloc[(page - 1) * page_size : page * page_size]

    if dataset == "players":
        rows = [{
            "player": r.player,
            "team": r.team,
            "goals_shots": f"{int(r.goals)}/{int(r.shots)}",
            "saves_faced": f"{int(r.saves)}/{int(r.shots_faced)}" if r.shots_faced > 0 else "–",
            "lnh_score": round(float(r.lnh_score), 1),
            "minutes": f"{int(r.seconds_played) // 60}:{int(r.seconds_played) % 60:02d}",
            "a": int(r.a),
            "br": int(r.br),
            "bp": int(r.bp),
            "2min": int(getattr(r, "2min", 0)),
            "opponent": r.opponent or "",
            "date": r.date,
            "season": r.season,
            "round": int(r.round),
        } for r in sub.itertuples()]
    else:
        rows = [{
            "date": r.date,
            "home_team": r.home_team,
            "score": r.score,
            "away_team": r.away_team,
            "ht_score": r.ht_score,
            "round": int(r.round),
            "season": r.season,
            "total_goals": int(r.total_goals) if r.total_goals >= 0 else "–",
            "venue": r.venue,
        } for r in sub.itertuples()]

    return {"dataset": dataset, "total": total, "page": page, "pages": pages, "rows": rows}


def api_export_csv(qs: dict) -> tuple[bytes, str]:
    dataset, df, full = _filter_explorer(qs)
    suffix = "complet" if full else "selection"
    fname = f"starligue_{dataset}_{suffix}.csv"
    if df.empty:
        return b"", fname
    if full:
        m, box, _ = load_data()
        raw = box if dataset == "players" else m
        return raw.to_csv(index=False).encode("utf-8-sig"), fname
    drop_tmp = [c for c in ["score", "ht_score", "total_goals", "player"] if c in df.columns]
    clean = df.drop(columns=drop_tmp)
    return clean.to_csv(index=False).encode("utf-8-sig"), fname


# ---- Simulations ------------------------------------------------------------
def run_job(jid: str, season: str, cutoff: int, n: int, keep_matches: bool):
    job = JOBS[jid]
    run_id = re.sub(r"[^0-9A-Za-z]+", "_", f"{season}_c{cutoff}_n{n}")
    out = OUT / run_id
    cmd = [sys.executable, str(HERE / "simulation_montecarlo.py"), "--data-dir", str(DATA), "--output-dir", str(out),
           "--season", season, "--cutoff-round", str(cutoff), "--n-simulations", str(n)]
    if not keep_matches:
        cmd.append("--no-export-matches")
    job.update(status="running", run=run_id, started=time.time())
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
        for line in p.stdout:
            job["log"].append(line.rstrip())
            job["log"] = job["log"][-60:]
        job["status"] = "done" if p.wait() == 0 else "error"
    except Exception as e:  # noqa
        job["log"].append(str(e)); job["status"] = "error"


def list_runs() -> list[dict]:
    runs = []
    for d in OUT.glob("*/summary_storytelling.json"):
        j = json.loads(d.read_text(encoding="utf-8"))
        runs.append({"id": d.parent.name, "season": j["season"], "cutoff": j["cutoff_round"], "n": j["n_simulations"],
                     "time": d.stat().st_mtime})
    return sorted(runs, key=lambda r: -r["time"])


def read_table(run: Path, stem: str) -> pd.DataFrame:
    pq, gz = run / f"{stem}.parquet", run / f"{stem}.csv.gz"
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(gz)


def run_dir(rid: str) -> Path:
    d = (OUT / re.sub(r"[^0-9A-Za-z_]", "", rid))
    if not d.is_dir():
        raise FileNotFoundError(rid)
    return d


def api_sim(rid: str) -> dict:
    d = run_dir(rid)
    j = json.loads((d / "summary_storytelling.json").read_text(encoding="utf-8"))
    ts = read_table(d, "team_summary").round(4).to_dict("records")
    ss = read_table(d, "sim_standings")
    notable = j["notable_simulations"]
    ids = {"typical": notable["typical_sim"]}
    for t, v in notable["by_team"].items():
        ids[f"best:{t}"], ids[f"worst:{t}"] = v["best_sim"], v["worst_sim"]
    pick = ss[ss["simulation"].isin(set(ids.values()))]
    tab = {int(k): g.sort_values("Rank")[["Team", "Rank", "Pts", "W", "D", "L", "GF", "GA", "GD"]].astype({"Team": str}).to_dict("records")
           for k, g in pick.groupby("simulation")}
    exp = pd.Series({r["Team"]: r["Expected_Pts"] for r in ts})
    piv = ss.assign(Team=ss["Team"].astype(str)).pivot(index="simulation", columns="Team", values="Pts")
    out_id = int(((piv - exp[piv.columns]) ** 2).sum(axis=1).idxmax())
    if out_id not in tab:
        tab[out_id] = ss[ss.simulation == out_id].sort_values("Rank").astype({"Team": str})[["Team", "Rank", "Pts", "W", "D", "L", "GF", "GA", "GD"]].to_dict("records")
    return {"season": j["season"], "cutoff": j["cutoff_round"], "n": j["n_simulations"], "teams": j["teams"],
            "summary": ts, "ratings": j.get("ratings_dc", []), "ids": ids, "outlier": out_id, "tables": tab,
            "rank_probs": j["rank_probabilities"], "metrics": j.get("backtest_metrics", {}),
            "rank_metrics": j.get("rank_metrics", {})}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, obj, code=200, ctype="application/json", headers: dict | None = None):
        b = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False, default=float).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path); parts = [x for x in u.path.split("/") if x]
        try:
            if not parts:
                return self.send((HERE / "index.html").read_bytes(), ctype="text/html")
            if parts[:2] == ["api", "meta"]: return self.send(meta())
            if parts[:2] == ["api", "history"]:
                if "history" not in _cache or _cache.get("h_stamp") != _cache.get("stamp"):
                    load_data(); _cache["history"] = build_history(); _cache["h_stamp"] = _cache["stamp"]
                return self.send(_cache["history"])
            if parts[:2] == ["api", "club"]:
                return self.send(api_club_detail(parse_qs(u.query)))
            if parts[:2] == ["api", "player"]:
                return self.send(api_player_detail(parse_qs(u.query)))
            if parts[:2] == ["api", "runs"]: return self.send(list_runs())
            if parts[:2] == ["api", "jobs"]: return self.send(JOBS.get(parts[2], {"status": "unknown"}))
            if parts[:2] == ["api", "sim"]:
                if len(parts) > 3 and parts[3] == "upcoming":
                    return self.send((run_dir(parts[2]) / "upcoming_scores.json").read_bytes())
                return self.send(api_sim(parts[2]))
            if parts[:2] == ["api", "explorer"]:
                return self.send(api_explorer(parse_qs(u.query)))
            if parts[:2] == ["api", "export"]:
                csv_bytes, fname = api_export_csv(parse_qs(u.query))
                return self.send(csv_bytes, ctype="text/csv", headers={"Content-Disposition": f'attachment; filename="{fname}"'})
            self.send({"error": "not found"}, 404)
        except FileNotFoundError as e:
            self.send({"error": f"Fichier introuvable : {e}"}, 404)
        except Exception as e:  # noqa
            self.send({"error": str(e)}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/simulate":
            return self.send({"error": "not found"}, 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        try:
            season, cutoff = str(body["season"]), max(0, int(body.get("cutoff_round", 0)))
            n = int(np.clip(int(body.get("n_simulations", 5000)), 200, 50000))
        except Exception:
            return self.send({"error": "Paramètres invalides"}, 400)
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"status": "queued", "log": [], "run": None}
        threading.Thread(target=run_job, args=(jid, season, cutoff, n, bool(body.get("keep_matches"))), daemon=True).start()
        self.send({"job": jid})


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA.is_dir():
        sys.exit(f"Dossier de données introuvable : {DATA}")
    print(f"Données : {DATA}\nSorties : {OUT}\n→ http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()