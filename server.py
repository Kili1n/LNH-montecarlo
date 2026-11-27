#!/usr/bin/env python3
"""Mini serveur LNH Starligue : Dashboard historique + Monte Carlo (stdlib + pandas).
Lancer :  python server.py   puis ouvrir http://localhost:8000"""
from __future__ import annotations

import argparse, contextlib, gzip, hashlib, io, json, math, os, re, shutil, subprocess, sys, threading, time, unicodedata, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

import simulation_montecarlo as sim

# ---- Configuration (en dur côté backend) -----------------------------------
DATA_DIR = "./data/data_csv"     # contient un sous-dossier par saison : match.csv, players.csv, players_boxscore.csv
OUTPUT_DIR = "./output"          # un sous-dossier par simulation lancée
CACHE_DIR = "./cache"            # bundles JSON prêts à afficher : À VERSIONNER (git) pour un hébergement à disque éphémère
CACHE_SIMS = int(os.environ.get("CACHE_SIMS", 10000))   # simulations par bundle
# Cache : saison en cours simulée depuis l'AVANT-DERNIÈRE journée jouée (K-1), pour pouvoir comparer les prédictions aux résultats de la
# dernière journée ; saison précédente depuis la J0.
# Garde-fous mémoire (hébergements plafonnés, ex. 512 Mo) -----------------------------------------
MAX_SIMS = int(os.environ.get("MAX_SIMS", 10000))             # plafond du nombre de simulations demandé depuis le site
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", 240))         # mémoire libre exigée avant de lancer une simulation (~220 Mo)
MEMORY_LIMIT_MB = int(os.environ.get("MEMORY_LIMIT_MB", 0))   # plafond de l'hébergeur si non détecté automatiquement
AUTO_BUILD_CACHE = os.environ.get("AUTO_BUILD_CACHE") == "1"  # génération du cache au démarrage : désactivée par défaut
ALLOW_KEEP_MATCHES = os.environ.get("ALLOW_KEEP_MATCHES") == "1"  # export sim_matches (≈ 900 Mo !) : désactivé sur le web
PORT = int(os.environ.get("PORT", 8000))
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA, OUT, CACHE = (HERE / DATA_DIR).resolve(), (HERE / OUTPUT_DIR).resolve(), (HERE / CACHE_DIR).resolve()
JOBS: dict[str, dict] = {}
_cache: dict = {}
_lock = threading.RLock()        # le serveur est multi-thread : on protège les caches

def _stamp() -> float:
    fs = list(DATA.rglob("*.csv"))
    return max((f.stat().st_mtime for f in fs), default=0.0)


_fp_memo: dict = {}


def data_fingerprint() -> str:
    """Empreinte du CONTENU des CSV (pas de leurs dates : un déploiement les réécrit toutes).
    Sert à savoir si le cache JSON correspond bien aux données actuelles. Recalculée seulement si un fichier change."""
    h = hashlib.sha1()
    for f in sorted(DATA.rglob("*.csv")):
        if f.name.startswith("_"):      # fichiers de travail du scraper (ex. _cache_players.csv) : sans effet sur les simulations
            continue
        st = f.stat()
        key = (str(f), st.st_mtime_ns, st.st_size)
        if key not in _fp_memo:
            raw = f.read_bytes().replace(b"\r\n", b"\n")   # insensible CRLF (Windows) / LF (Linux, Render)
            _fp_memo[key] = (len(raw), hashlib.sha1(raw).hexdigest())
        n_b, dig = _fp_memo[key]
        h.update(f"{f.relative_to(DATA).as_posix()}:{n_b}:{dig};".encode())
    return h.hexdigest()[:16]


def _read_int(path: str) -> int | None:
    try:
        v = Path(path).read_text().strip()
        return None if v == "max" else int(v)
    except Exception:
        return None


def mem_limit_mb() -> float | None:
    """Plafond mémoire du conteneur : variable MEMORY_LIMIT_MB, sinon cgroup v2 puis v1."""
    if MEMORY_LIMIT_MB:
        return float(MEMORY_LIMIT_MB)
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        v = _read_int(p)
        if v and v < (1 << 45):
            return v / 1e6
    return None


def mem_used_mb() -> float | None:
    """Mémoire « de travail » du conteneur (hors cache disque récupérable), comme le calcule l'hébergeur."""
    for cur_p, stat_p, key in (("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", "inactive_file"),
                               ("/sys/fs/cgroup/memory/memory.usage_in_bytes", "/sys/fs/cgroup/memory/memory.stat", "total_inactive_file")):
        cur = _read_int(cur_p)
        if cur is not None:
            inactive = 0
            try:
                for line in Path(stat_p).read_text().splitlines():
                    k, _, v = line.partition(" ")
                    if k == key:
                        inactive = int(v)
            except Exception:
                pass
            return (cur - inactive) / 1e6
    try:  # sans cgroup : mémoire résidente de ce processus et de ses enfants directs
        total = 0.0
        me = os.getpid()
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                st = (d / "status").read_text()
                ppid = int(re.search(r"PPid:\s+(\d+)", st).group(1))
                if int(d.name) == me or ppid == me:
                    total += int(re.search(r"VmRSS:\s+(\d+)", st).group(1)) / 1024
            except Exception:
                continue
        return total or None
    except Exception:
        return None


def mem_available_mb() -> float | None:
    lim, used = mem_limit_mb(), mem_used_mb()
    return None if lim is None or used is None else lim - used


def _fold(s: str) -> str:
    """Minuscule sans accents (recherche de joueurs)."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _jsonable(o):
    """Rend n'importe quelle structure sérialisable en JSON valide (NaN/inf -> null, types numpy -> natifs)."""
    if o is None or isinstance(o, (str, bool)):
        return o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = round(float(o), 6) if isinstance(o, np.float32) else float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, pd.DataFrame):
        return _jsonable(o.to_dict("records"))
    if o is pd.NaT:
        return None
    return str(o)


def load_matches():
    """Matchs seuls (quelques milliers de lignes) : suffit à l'onglet Monte Carlo, sans charger les feuilles de match."""
    with _lock:
        st = _stamp()
        if _cache.get("mstamp") == st and "m_only" in _cache:
            return _cache["m_only"], _cache["seasons_only"]
        amap = sim.load_team_aliases(DATA)
        m, seasons, _ = sim.load_all_matches(DATA, None, alias_map=amap)
        _cache.update(mstamp=st, m_only=m, seasons_only=seasons)
        return m, seasons


# Feuilles de match LNH : « DP » = Dernière Passe (passe décisive), « A » = Avertissement (carton jaune), « D » = Disqualification.
# La LNH ne relève les passes décisives de façon complète qu'à partir de 2023/24 (≈ 45 par match contre ≈ 6 avant, 0 avant 2013) :
# les classements de passes ne portent donc que sur ces saisons, pour ne pas comparer l'incomparable.
DP_RELIABLE_FROM = "2023/2024"


def _iso_date(v) -> str:
    """Date ISO « AAAA-MM-JJ » ou chaîne vide. Dans les CSV (match.csv) la date est toujours AAAA-MM-JJ ou absente (matchs à venir non
    programmés) ; une valeur manquante devenait la chaîne « nan », que le navigateur affichait « Invalid Date »."""
    s = str(v).strip()
    return s[:10] if re.match(r"^\d{4}-\d{2}-\d{2}", s) else ""


def _add_match_context(box: pd.DataFrame, m: pd.DataFrame) -> None:
    """Ajoute à la feuille de match (EN PLACE, sans copie) la date, la journée, l'adversaire et le score."""
    mi = m.drop_duplicates("match_id").set_index("match_id")
    mid = box["match_id"]
    home, away = mid.map(mi["home_team"]), mid.map(mi["away_team"])
    is_home = (box["team"] == home).to_numpy()
    box["opponent"] = np.where(is_home, away, home)
    hg = pd.to_numeric(mid.map(mi["home_ft_goals"]), errors="coerce")
    ag = pd.to_numeric(mid.map(mi["away_ft_goals"]), errors="coerce")
    gf = pd.Series(np.where(is_home, hg, ag), index=box.index)
    ga = pd.Series(np.where(is_home, ag, hg), index=box.index)
    box["score"] = np.where(gf.notna(), gf.fillna(0).astype(int).astype(str) + "–" + ga.fillna(0).astype(int).astype(str), "")
    box["date"] = mid.map(mi["date"]).fillna("").astype(str)
    box["round"] = pd.to_numeric(mid.map(mi["round"]), errors="coerce").fillna(0).astype(np.int16)
    box["season_idx"] = pd.to_numeric(mid.map(mi["season_idx"]), errors="coerce").fillna(-1).astype(np.int16)
    box["two_min"] = box["2min"]
    box["ast_ok"] = (box["season"].astype(str) >= DP_RELIABLE_FROM).to_numpy()   # passes décisives fiables sur cette saison ?
    box["ast"] = np.where(box["ast_ok"], box["dp"], 0).astype(np.int32)         # passes décisives (DP)
    box["yc"] = box["a"]                                                        # avertissements (cartons jaunes)
    box["rc"] = box["d"]                                                        # disqualifications (cartons rouges)


def load_data():
    """Charge matchs + feuilles de match une seule fois (rechargé si les CSV changent)."""
    with _lock:
        st = _stamp()
        if _cache.get("stamp") == st and "m" in _cache:
            return _cache["m"], _cache["box"], _cache["seasons"]
        for k in [k for k in _cache if k not in ("stamp", "mstamp", "m_only", "seasons_only")]:
            _cache.pop(k, None)
        amap = sim.load_team_aliases(DATA)
        has_box = bool(sim._find_csvs(DATA, "players_boxscore.csv"))
        box = sim.load_player_tables(DATA, DATA, amap) if has_box else None
        m, seasons = load_matches()
        if box is not None and not m.empty:
            season_by_match = m.drop_duplicates("match_id").set_index("match_id")["season"]
            mapped = box["match_id"].map(season_by_match)
            box["season"] = mapped.fillna(box["season"].astype(str).str.replace("-", "/", regex=False))
            box = sim.add_player_ratings(box, m)          # en place
            box["full_name"] = (box["first_name"].fillna("").astype(str) + " " + box["last_name"].fillna("").astype(str)).str.strip()
            _add_match_context(box, m)
        _cache.update(stamp=st, m=m, box=box, seasons=seasons)
        sim.trim_memory()
        return m, box, seasons


_PLAYED_COLS = ["match_id", "full_name", "team", "opponent", "date", "score", "season", "season_idx", "round", "is_gk", "goals", "shots",
                "goals_7m", "shots_7m", "saves", "shots_faced", "saves_7m", "shots_faced_7m", "ast", "ast_ok", "yc", "rc", "br", "tc", "bp", "2min", "match_note",
                "def_note", "impact_def", "est_seconds"]


def played_rows() -> pd.DataFrame:
    """Lignes joueur-match réellement jouées, limitées aux colonnes utiles (~15 Mo au lieu de ~30 Mo), mises en cache."""
    load_data()
    with _lock:
        if "played_rows" not in _cache:
            pf = player_frame()
            _cache["played_rows"] = pf.loc[pf["played_match"], _PLAYED_COLS] if pf is not None else pd.DataFrame(columns=_PLAYED_COLS)
        return _cache["played_rows"]


def player_frame() -> pd.DataFrame | None:
    """Feuilles de match enrichies (date, journée, adversaire, score). C'est la même table que `box` : aucune copie."""
    m, box, _ = load_data()
    return box if box is not None and not box.empty else None


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
    empty = {"mvp": [], "scorers": [], "keepers": [], "seven_m": [], "defense": [], "assists": [], "discipline": []}
    if box is None:
        return empty
    b = box if season is None else box[box["season"] == season]
    b = b[b["played_match"]]
    if b.empty:
        return empty
    g = b.groupby(["full_name", "team"]).agg(
        games=("match_id", "nunique"),
        seconds=("est_seconds", "sum"),
        real_clock_share=("has_real_clock", "mean"),
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
        ast=("ast", "sum"),
        yc=("yc", "sum"),
        rc=("rc", "sum"),
        bp=("bp", "sum"),
        two_min=("2min", "sum"),
        impact=("impact_score", "sum"),
        note=("match_note", "mean"),
        def_note=("def_note", "mean"),
        impact_def=("impact_def", "sum"),
        gk=("is_gk", "max"),
    ).reset_index().rename(columns={"full_name": "name"})

    g["note"] = g["note"].fillna(5.0).round(2)
    g["def_note"] = g["def_note"].fillna(5.0).round(2)
    g["impact_def"] = g["impact_def"].round(1)
    g["impact"] = g["impact"].round(1)
    g["est_clock"] = g["real_clock_share"] < 0.5

    g["minutes"] = (g["seconds"] / 60.0).round(1)
    g["pct"] = np.where(g.shots > 0, 100 * g.goals / g.shots.clip(lower=1), 0).round(1)
    g["save_pct"] = np.where(g.faced > 0, 100 * g.saves / g.faced.clip(lower=1), 0).round(1)
    g["pct_7m"] = np.where(g.shots_7m > 0, 100 * g.goals_7m / g.shots_7m.clip(lower=1), 0).round(1)
    g["def_act"] = g["br"] + g["tc"]
    g["sanctions"] = g["yc"] + g["two_min"] + g["rc"]   # discipline : avertissements + 2 minutes + disqualifications

    cols = [
        "name", "team", "games", "minutes", "goals", "shots", "pct",
        "goals_7m", "shots_7m", "pct_7m", "saves", "faced", "save_pct",
        "saves_7m", "faced_7m", "br", "tc", "def_act", "ast", "yc", "rc", "sanctions", "bp", "two_min",
        "impact", "note", "def_note", "impact_def", "est_clock", "gk",
    ]
    rec = lambda d: d[cols].to_dict("records")
    min_faced = 60 if season else 150
    min_7m = 10 if season else 25
    min_games_mvp = 5 if season else 15
    min_games_def = 5 if season else 15
    mvp_pool = g[g["games"] >= min_games_mvp]
    if len(mvp_pool) < 5:
        mvp_pool = g
    def_pool = g[(~g.gk) & (g["games"] >= min_games_def)]
    if len(def_pool) < 5:
        def_pool = g[~g.gk]
    return {
        "mvp": rec(mvp_pool.sort_values(["note", "impact"], ascending=False).head(40)),
        "scorers": rec(g[~g.gk].sort_values("goals", ascending=False).head(40)),
        "keepers": rec(g[g.gk & (g.faced >= min_faced)].sort_values("save_pct", ascending=False).head(40)),
        "seven_m": rec(g[g.shots_7m >= min_7m].sort_values(["goals_7m", "pct_7m"], ascending=False).head(40)),
        # nouvelle métrique défensive : points d'impact défensif (récupérations, contres, bloc collectif, discipline)
        "defense": rec(def_pool.sort_values(["impact_def", "def_note"], ascending=False).head(40)),
        "assists": rec(g[(~g.gk) & (g["ast"] > 0)].sort_values(["ast", "games"], ascending=[False, True]).head(40)),   # saisons où la DP est relevée
        "discipline": rec(g[~g.gk].sort_values(["sanctions", "yc", "two_min"], ascending=False).head(40)),
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
    m, seasons = load_matches()
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
            "match_id": str(r.match_id),
            "date": str(r.date),
            "round": int(r.round),
            "season": str(r.season),
            "is_home": bool(is_h),
            "opponent": str(r.away_team if is_h else r.home_team),
            "score": f"{gf}–{ga}",
            "res": res,
        })

    # Effectif sur la période (note LNH /10 et note défensive)
    roster = []
    if box is not None and not box.empty:
        b_club = _filter_seasons_df(box[(box["team"] == team) & box["played_match"]], seasons_raw)
        if not b_club.empty:
            rg = b_club.groupby("full_name").agg(
                games=("match_id", "nunique"),
                seconds=("est_seconds", "sum"),
                goals=("goals", "sum"),
                shots=("shots", "sum"),
                saves=("saves", "sum"),
                faced=("shots_faced", "sum"),
                note=("match_note", "mean"),
                impact_def=("impact_def", "sum"),
                gk=("is_gk", "max"),
            ).reset_index().sort_values(["seconds", "impact_def"], ascending=False)
            roster = [{
                "name": str(r.full_name),
                "games": int(r.games),
                "min_m": round(float(r.seconds) / 60.0 / max(int(r.games), 1), 1),
                "goals": int(r.goals),
                "shots": int(r.shots),
                "saves": int(r.saves),
                "faced": int(r.faced),
                "note": round(float(r.note), 2) if pd.notna(r.note) else None,
                "impact_def": round(float(r.impact_def), 1),
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
        "roster": roster,
        "history": hist_ranks,
    }


# ---- Métriques d'équipes (scatter à axes configurables) ---------------------
def api_teams(qs: dict) -> dict:
    """Une ligne par club sur la période : ~30 indicateurs que le scatter peut croiser librement."""
    seasons_raw = (qs.get("seasons", ["all"])[0] or "all").strip()
    key = ("teams", seasons_raw)
    with _lock:
        if key in _cache:
            return _cache[key]
    m, box, _ = load_data()
    mp = _filter_seasons_df(m[m["played"]], seasons_raw)
    if mp.empty:
        return {"teams": [], "period": seasons_raw}

    hm = pd.DataFrame({"t": mp["home_team"], "gf": mp["home_ft_goals"], "ga": mp["away_ft_goals"], "home": 1})
    aw = pd.DataFrame({"t": mp["away_team"], "gf": mp["away_ft_goals"], "ga": mp["home_ft_goals"], "home": 0})
    x = pd.concat([hm, aw], ignore_index=True)
    x["w"] = (x.gf > x.ga).astype(int)
    x["d"] = (x.gf == x.ga).astype(int)
    x["l"] = (x.gf < x.ga).astype(int)
    x["close"] = ((x.gf - x.ga).abs() <= 2).astype(int)
    x["w_h"], x["w_a"] = x.w * x.home, x.w * (1 - x.home)
    g = x.groupby("t").agg(j=("gf", "size"), w=("w", "sum"), d=("d", "sum"), l=("l", "sum"),
                           gf=("gf", "sum"), ga=("ga", "sum"), close=("close", "sum"),
                           j_h=("home", "sum"), w_h=("w_h", "sum"), w_a=("w_a", "sum"))
    g["pts"] = 2 * g.w + g.d
    g["j_a"] = g.j - g.j_h
    out = pd.DataFrame({
        "team": g.index.astype(str),
        "j": g.j, "pts": g.pts,
        "pts_m": g.pts / g.j,
        "win_pct": 100 * g.w / g.j,
        "draw_pct": 100 * g.d / g.j,
        "gf_m": g.gf / g.j, "ga_m": g.ga / g.j,
        "gd_m": (g.gf - g.ga) / g.j,
        "tot_m": (g.gf + g.ga) / g.j,
        "close_pct": 100 * g.close / g.j,
        "home_win_pct": np.where(g.j_h > 0, 100 * g.w_h / g.j_h.clip(lower=1), np.nan),
        "away_win_pct": np.where(g.j_a > 0, 100 * g.w_a / g.j_a.clip(lower=1), np.nan),
    }).reset_index(drop=True)
    st = standings(mp)[["t", "rank"]].rename(columns={"t": "team"})
    out = out.merge(st, on="team", how="left")

    ht = pd.DataFrame(halftime_stats(mp))
    if not ht.empty:
        ht["h2_diff"] = (ht["h2_gf"] - ht["h2_ga"]) / ht["j_ht"].clip(lower=1)
        ht["kept_pct"] = np.where(ht["ahead_ht"] > 0, 100 * ht["kept"] / ht["ahead_ht"].clip(lower=1), np.nan)
        ht["cb_pct"] = np.where(ht["behind_ht"] > 0, 100 * (ht["cb_w"] + ht["cb_d"]) / ht["behind_ht"].clip(lower=1), np.nan)
        out = out.merge(ht[["team", "cb_pts", "h2_diff", "kept_pct", "cb_pct"]], on="team", how="left")

    pf = player_frame()
    if pf is not None:
        pb = _filter_seasons_df(played_rows(), seasons_raw)
        if not pb.empty:
            tg = pb.groupby("team").agg(
                bx_games=("match_id", "nunique"), goals=("goals", "sum"), shots=("shots", "sum"),
                g7=("goals_7m", "sum"), s7=("shots_7m", "sum"), saves=("saves", "sum"), faced=("shots_faced", "sum"),
                ast=("ast", "sum"), yc=("yc", "sum"), bp=("bp", "sum"), br=("br", "sum"), tc=("tc", "sum"), two=("2min", "sum"),
            )
            ast_games = pb[pb["ast_ok"]].groupby("team")["match_id"].nunique().reindex(tg.index)   # passes décisives : saisons fiables seulement
            core = pb[pb["est_seconds"] >= 600]
            notes = core.groupby("team").agg(note=("match_note", "mean"), def_note=("def_note", "mean"))
            ex = pd.DataFrame({
                "shoot_pct": 100 * tg.goals / tg.shots.clip(lower=1),
                "pct_7m": np.where(tg.s7 > 0, 100 * tg.g7 / tg.s7.clip(lower=1), np.nan),
                "save_pct": np.where(tg.faced > 0, 100 * tg.saves / tg.faced.clip(lower=1), np.nan),
                "ast_m": tg.ast / ast_games.where(ast_games > 0), "yc_m": tg.yc / tg.bx_games, "bp_m": tg.bp / tg.bx_games,
                "br_m": tg.br / tg.bx_games, "tc_m": tg.tc / tg.bx_games,
                "two_min_m": tg.two / tg.bx_games,
            }, index=tg.index).join(notes).reset_index().rename(columns={"index": "team"})
            ex = ex.rename(columns={ex.columns[0]: "team"})
            out = out.merge(ex, on="team", how="left")

    res = {"period": seasons_raw, "teams": _jsonable(out.round(3))}
    with _lock:
        _cache[key] = res
    return res


# ---- Records ----------------------------------------------------------------
def _since() -> str:
    """« depuis 2007 » : première saison couverte par les données. La base ne remonte pas à l'origine de la Starligue : tout total ou
    record affiché doit le préciser, pour ne pas passer pour un total de carrière."""
    _, _, seasons = load_data()
    return f"depuis {str(seasons[0])[:4]}" if seasons else ""


def _team_calendar() -> dict:
    """(club, match_id) -> (rang du match dans le calendrier du club, 1er de sa saison ?, dernier de sa saison ?, season_idx, journée).
    Calendrier = TOUS les matchs joués du club en Starligue (avec ou sans feuille de match)."""
    m, _, _ = load_data()
    with _lock:
        if "team_cal" in _cache:
            return _cache["team_cal"]
    mp = m[m["played"]]
    c = pd.concat([
        mp[["match_id", "season_idx", "round", "date", "home_team"]].rename(columns={"home_team": "team"}),
        mp[["match_id", "season_idx", "round", "date", "away_team"]].rename(columns={"away_team": "team"}),
    ], ignore_index=True).sort_values(["team", "season_idx", "round", "date"], kind="mergesort")
    c["tidx"] = c.groupby("team").cumcount()
    c["first"] = c.groupby(["team", "season_idx"]).cumcount() == 0
    c["last"] = c.groupby(["team", "season_idx"]).cumcount(ascending=False) == 0
    cal = {(str(t), str(mid)): (int(i), bool(f), bool(l), int(si), int(r))
           for t, mid, i, f, l, si, r in zip(c["team"], c["match_id"], c["tidx"], c["first"], c["last"], c["season_idx"], c["round"])}
    with _lock:
        _cache["team_cal"] = cal
    return cal


def _linked(prev, cur, same_team: bool) -> bool:
    """Deux apparitions consécutives d'un joueur sont-elles deux matchs consécutifs de son équipe, sans saison hors Starligue entre les deux ?"""
    if prev is None or cur is None or cur[3] - prev[3] > 1:
        return False
    if same_team:
        return cur[0] == prev[0] + 1
    return prev[2] and cur[1] and cur[3] == prev[3] + 1      # changement de club : fin de saison -> début de la suivante


def _longest_scoring_run(grp: pd.DataFrame, cal: dict) -> tuple[int, int, int]:
    """Plus longue série de matchs consécutifs AVEC un but. Elle casse quand le joueur manque un match de son équipe
    (blessure, banc, absence de feuille) ou quitte la Starligue entre deux saisons."""
    teams, mids, goals = grp["team"].astype(str).tolist(), grp["match_id"].astype(str).tolist(), grp["goals"].to_numpy()
    best, cur, start, prev, prev_team = (0, -1, -1), 0, 0, None, None
    for i in range(len(goals)):
        info = cal.get((teams[i], mids[i]))
        if goals[i] > 0:
            if cur > 0 and _linked(prev, info, teams[i] == prev_team):
                cur += 1
            else:
                cur, start = 1, i
            if cur > best[0]:
                best = (cur, start, i)
        else:
            cur = 0
        prev, prev_team = info, teams[i]
    return best


def _current_scoring_run(grp: pd.DataFrame, cal: dict, team: str, team_last_idx: int) -> int:
    """Série EN COURS : le joueur doit avoir joué (et marqué lors de) le tout dernier match de son équipe."""
    teams, mids, goals = grp["team"].astype(str).tolist(), grp["match_id"].astype(str).tolist(), grp["goals"].to_numpy()
    if not len(goals) or teams[-1] != team:
        return 0
    info = cal.get((teams[-1], mids[-1]))
    if info is None or info[0] != team_last_idx:
        return 0
    n = 0
    for i in range(len(goals) - 1, -1, -1):
        if goals[i] <= 0:
            break
        n += 1
        if i == 0:
            break
        if not _linked(cal.get((teams[i - 1], mids[i - 1])), cal.get((teams[i], mids[i])), teams[i - 1] == teams[i]):
            break
    return n


def _longest_run(flags) -> tuple[int, int, int]:
    """(longueur, index de début, index de fin) de la plus longue série de True."""
    best, cur, start = (0, -1, -1), 0, 0
    for i, f in enumerate(flags):
        if f:
            if cur == 0:
                start = i
            cur += 1
            if cur > best[0]:
                best = (cur, start, i)
        else:
            cur = 0
    return best


def _fmtv(v: float, d: int = 0) -> str:
    v = float(v)
    return str(int(round(v))) if d == 0 else f"{v:.{d}f}".replace(".", ",")


def _row(holder, value, detail="", *, team="", season="", date="", kind="team", d=0) -> dict:
    return {"holder": str(holder), "team": str(team), "value": float(value), "display": _fmtv(value, d),
            "detail": detail, "season": str(season), "date": str(date), "kind": kind}


def _team_match_table(mp: pd.DataFrame) -> pd.DataFrame:
    nan = pd.Series(np.nan, index=mp.index)
    hth = pd.to_numeric(mp["home_ht_goals"], errors="coerce") if "home_ht_goals" in mp.columns else nan
    ath = pd.to_numeric(mp["away_ht_goals"], errors="coerce") if "away_ht_goals" in mp.columns else nan

    def side(home: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "match_id": mp["match_id"], "season": mp["season"], "season_idx": mp["season_idx"],
            "round": mp["round"], "date": mp["date"].astype(str),
            "team": mp["home_team"] if home else mp["away_team"],
            "opp": mp["away_team"] if home else mp["home_team"],
            "home": home,
            "gf": mp["home_ft_goals"] if home else mp["away_ft_goals"],
            "ga": mp["away_ft_goals"] if home else mp["home_ft_goals"],
            "ht_gf": hth if home else ath, "ht_ga": ath if home else hth,
        })
    tm = pd.concat([side(True), side(False)], ignore_index=True)
    tm["margin"] = tm["gf"] - tm["ga"]
    tm["total"] = tm["gf"] + tm["ga"]
    tm["res"] = np.where(tm["margin"] > 0, "W", np.where(tm["margin"] == 0, "D", "L"))
    return tm.sort_values(["season_idx", "round", "date"], kind="mergesort").reset_index(drop=True)


def compute_records(seasons_raw: str = "all") -> dict:
    seasons_raw = (seasons_raw or "all").strip() or "all"
    key = ("records", seasons_raw)
    with _lock:
        if key in _cache:
            return _cache[key]
    m, box, _ = load_data()
    mp = _filter_seasons_df(m[m["played"]], seasons_raw)
    groups: list[dict] = []
    N = 5

    # ---- 1. Matchs & équipes -------------------------------------------------
    if not mp.empty:
        tm = _team_match_table(mp)
        loc = lambda r: ("vs" if r.home else "@")
        recs = []

        d = tm[tm["margin"] > 0].sort_values(["margin", "gf", "date"], ascending=[False, False, True]).head(N)
        recs.append({"id": "biggest_win", "label": "Plus large victoire", "unit": "buts d'écart", "top": [
            _row(r.team, r.margin, f"{int(r.gf)}–{int(r.ga)} {loc(r)} {r.opp} · J{int(r.round)}", season=r.season, date=r.date) for r in d.itertuples()]})

        hmatch = tm[tm["home"]]
        d = hmatch.sort_values(["total", "date"], ascending=[False, True]).head(N)
        recs.append({"id": "highest_total", "label": "Match le plus prolifique", "unit": "buts au total", "top": [
            _row(f"{r.team} – {r.opp}", r.total, f"{int(r.gf)}–{int(r.ga)} · J{int(r.round)}", season=r.season, date=r.date, kind="match") for r in d.itertuples()]})
        d = hmatch.sort_values(["total", "date"], ascending=[True, True]).head(N)
        recs.append({"id": "lowest_total", "label": "Match le plus fermé", "unit": "buts au total", "top": [
            _row(f"{r.team} – {r.opp}", r.total, f"{int(r.gf)}–{int(r.ga)} · J{int(r.round)}", season=r.season, date=r.date, kind="match") for r in d.itertuples()]})

        d = tm.sort_values(["gf", "margin", "date"], ascending=[False, False, True]).head(N)
        recs.append({"id": "most_scored", "label": "Plus de buts marqués par une équipe", "unit": "buts", "top": [
            _row(r.team, r.gf, f"{int(r.gf)}–{int(r.ga)} {loc(r)} {r.opp} · J{int(r.round)}", season=r.season, date=r.date) for r in d.itertuples()]})

        dr = hmatch[hmatch["res"] == "D"].sort_values(["total", "date"], ascending=[False, True]).head(N)
        recs.append({"id": "wild_draw", "label": "Nul le plus spectaculaire", "unit": "buts au total", "top": [
            _row(f"{r.team} – {r.opp}", r.total, f"{int(r.gf)}–{int(r.ga)} · J{int(r.round)}", season=r.season, date=r.date, kind="match") for r in dr.itertuples()]})

        ht = tm[tm["ht_gf"].notna() & ((tm["ht_gf"] + tm["ht_ga"]) > 0)].copy()
        if not ht.empty:
            ht["lead"] = ht["ht_gf"] - ht["ht_ga"]
            d = ht.sort_values(["lead", "date"], ascending=[False, True]).head(N)
            recs.append({"id": "ht_lead", "label": "Plus grosse avance à la pause", "unit": "buts d'écart", "top": [
                _row(r.team, r.lead, f"{int(r.ht_gf)}–{int(r.ht_ga)} à la pause, final {int(r.gf)}–{int(r.ga)} vs {r.opp}", season=r.season, date=r.date) for r in d.itertuples()]})
            cb = ht[(ht["lead"] < 0) & (ht["margin"] > 0)].assign(deficit=lambda z: -z["lead"])
            d = cb.sort_values(["deficit", "margin", "date"], ascending=[False, False, True]).head(N)
            recs.append({"id": "comeback", "label": "Plus grande remontada", "unit": "buts de retard effacés", "top": [
                _row(r.team, r.deficit, f"menée {int(r.ht_gf)}–{int(r.ht_ga)} à la pause, gagne {int(r.gf)}–{int(r.ga)} {loc(r)} {r.opp}", season=r.season, date=r.date) for r in d.itertuples()]})
            ht["h2"] = (ht["gf"] - ht["ht_gf"]) - (ht["ga"] - ht["ht_ga"])
            d = ht.sort_values(["h2", "date"], ascending=[False, True]).head(N)
            recs.append({"id": "best_half", "label": "Meilleure seconde période", "unit": "buts d'écart en 2e mi-temps", "top": [
                _row(r.team, r.h2, f"{int(r.gf - r.ht_gf)}–{int(r.ga - r.ht_ga)} en 2e période {loc(r)} {r.opp}", season=r.season, date=r.date) for r in d.itertuples()]})
        groups.append({"id": "matches", "title": "Matchs", "blurb": "Les scores qui ont marqué la Starligue.", "records": recs})

        # ---- 2. Séries (au sein d'une même saison) ---------------------------
        def streak_rows(cond, home_only=False, label="") -> list[dict]:
            best = []
            for (team, season), grp in tm.groupby(["team", "season"], sort=False):
                grp = grp[grp["home"]] if home_only else grp
                if grp.empty:
                    continue
                ln, a, b = _longest_run(cond(grp).to_numpy())
                if ln > 0:
                    best.append((ln, team, season, int(grp["round"].iloc[a]), int(grp["round"].iloc[b]), str(grp["date"].iloc[b])))
            best.sort(key=lambda t: (-t[0], t[5]))
            return [_row(t[1], t[0], f"{t[2]} · de la J{t[3]} à la J{t[4]}", season=t[2], date=t[5]) for t in best[:N]]

        groups.append({"id": "streaks", "title": "Séries", "blurb": "Calculées au sein d'une même saison.", "records": [
            {"id": "win_streak", "label": "Plus longue série de victoires", "unit": "matchs", "top": streak_rows(lambda z: z["res"] == "W")},
            {"id": "unbeaten", "label": "Plus longue série d'invincibilité", "unit": "matchs", "top": streak_rows(lambda z: z["res"] != "L")},
            {"id": "home_unbeaten", "label": "Plus longue série invaincue à domicile", "unit": "matchs", "top": streak_rows(lambda z: z["res"] != "L", home_only=True)},
            {"id": "loss_streak", "label": "Plus longue série de défaites", "unit": "matchs", "top": streak_rows(lambda z: z["res"] == "L")},
        ]})

        # ---- 3. Saisons de club ---------------------------------------------
        ts = tm.groupby(["team", "season"]).agg(j=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum"),
                                                w=("res", lambda s: int((s == "W").sum())), d=("res", lambda s: int((s == "D").sum()))).reset_index()
        ts["pts"] = 2 * ts.w + ts.d
        ts["gd"] = ts.gf - ts.ga
        full = ts[ts["j"] >= 0.9 * ts["j"].max()]
        if len(full) >= 3:
            def top_ts(col, asc, label, unit, fmt=lambda r: f"{r.w}V {r.d}N {int(r.j - r.w - r.d)}D · {int(r.gf)}–{int(r.ga)}"):
                keys = [col] + [k for k in ("pts", "gd") if k != col]
                d_ = full.sort_values(keys, ascending=[asc] + [not asc] * (len(keys) - 1)).head(N)
                return {"id": f"ts_{col}_{'min' if asc else 'max'}", "label": label, "unit": unit,
                        "top": [_row(r.team, getattr(r, col), fmt(r), season=r.season) for r in d_.itertuples()]}
            groups.append({"id": "club_seasons", "title": "Saisons de club", "blurb": "Saisons complètes uniquement.", "records": [
                top_ts("pts", False, "Meilleur total de points", "points"),
                top_ts("gf", False, "Meilleure attaque", "buts marqués"),
                top_ts("ga", True, "Meilleure défense", "buts encaissés"),
                top_ts("gd", False, "Meilleure différence de buts", "buts"),
                top_ts("pts", True, "Saison la plus difficile", "points"),
            ]})

    # ---- 4. Joueurs : un seul match ------------------------------------------
    pf = player_frame()
    if pf is not None:
        pp = _filter_seasons_df(played_rows(), seasons_raw)
        if not pp.empty:
            pp = pp.assign(save_ratio=np.where(pp["shots_faced"] >= 15, 100 * pp["saves"] / pp["shots_faced"].clip(lower=1), np.nan))

            def p_top(col, label, unit, detail, *, d=0, mask=None, minv=0.0):
                df = pp if mask is None else pp[mask(pp)]
                df = df[df[col].notna() & (df[col] > minv)].sort_values([col, "date"], ascending=[False, True]).drop_duplicates("full_name").head(N)
                return {"id": f"pm_{col}", "label": label, "unit": unit, "top": [
                    _row(r.full_name, getattr(r, col), detail(r), team=r.team, season=r.season, date=r.date, kind="player", d=d) for r in df.itertuples()]}

            base = lambda r: f"vs {r.opponent} ({r.score}) · J{int(r.round)}"
            gk_mask = lambda z: z["is_gk"].astype(bool)
            groups.append({"id": "players_match", "title": "Joueurs : un seul match", "blurb": "Meilleure performance de chaque joueur (un nom n'apparaît qu'une fois par classement).", "records": [
                p_top("goals", "Plus de buts", "buts", lambda r: f"{int(r.goals)}/{int(r.shots)} aux tirs {base(r)}", mask=lambda z: ~z["is_gk"].astype(bool)),
                p_top("goals_7m", "Plus de jets de 7 m marqués", "buts à 7 m", lambda r: f"{int(r.goals_7m)}/{int(r.shots_7m)} {base(r)}"),
                p_top("ast", "Plus de passes décisives", "passes", base),
                p_top("yc", "Plus d'avertissements (cartons jaunes)", "avertissements", base),
                p_top("br", "Plus de balles récupérées", "BR", base),
                p_top("tc", "Plus de contres", "TC", base),
                p_top("saves", "Plus d'arrêts", "arrêts", lambda r: f"{int(r.saves)}/{int(r.shots_faced)} {base(r)}", mask=gk_mask),
                p_top("saves_7m", "Plus d'arrêts sur 7 m", "arrêts à 7 m", lambda r: f"{int(r.saves_7m)}/{int(r.shots_faced_7m)} {base(r)}", mask=gk_mask),
                p_top("save_ratio", "Meilleur % d'arrêts (15 tirs minimum)", "% d'arrêts", lambda r: f"{int(r.saves)}/{int(r.shots_faced)} {base(r)}", d=1, mask=gk_mask),
                p_top("match_note", "Meilleure note /10 sur un match", "/ 10", lambda r: f"{base(r)}", d=2),
            ]})

            # ---- 5. Joueurs : saison & série --------------------------------
            ps = pp.groupby(["full_name", "team", "season"]).agg(
                games=("match_id", "nunique"), goals=("goals", "sum"), g7=("goals_7m", "sum"), ast=("ast", "sum"), yc=("yc", "sum"),
                saves=("saves", "sum"), faced=("shots_faced", "sum"), br=("br", "sum"), tc=("tc", "sum"),
                note=("match_note", "mean"), impact_def=("impact_def", "sum"), gk=("is_gk", "max")).reset_index()
            ps["def_act"] = ps["br"] + ps["tc"]
            min_g = 8 if seasons_raw not in ("all",) and "," not in seasons_raw else 10

            def s_top(col, label, unit, detail, *, d=0, mask=None, minv=0.0):
                df = ps if mask is None else ps[mask(ps)]
                df = df[df[col].notna() & (df[col] > minv)].sort_values([col, "games"], ascending=[False, True]).head(N)
                return {"id": f"ps_{col}", "label": label, "unit": unit, "top": [
                    _row(r.full_name, getattr(r, col), detail(r), team=r.team, season=r.season, kind="player", d=d) for r in df.itertuples()]}

            sd = lambda r: f"{r.games} matchs · {r.season}"
            recs = [
                s_top("goals", "Plus de buts sur une saison", "buts", sd, mask=lambda z: ~z["gk"].astype(bool)),
                s_top("g7", "Plus de buts à 7 m sur une saison", "buts à 7 m", sd),
                s_top("ast", "Plus de passes décisives sur une saison", "passes", sd),
                s_top("yc", "Plus d'avertissements sur une saison", "avertissements", sd, mask=lambda z: ~z["gk"].astype(bool)),
                s_top("saves", "Plus d'arrêts sur une saison", "arrêts", sd, mask=lambda z: z["gk"].astype(bool)),
                s_top("def_act", "Plus de BR + TC sur une saison", "actions défensives", sd, mask=lambda z: ~z["gk"].astype(bool)),
                s_top("note", f"Meilleure note /10 moyenne ({min_g} matchs min.)", "/ 10", sd, d=2, mask=lambda z: z["games"] >= min_g),
            ]
            # série de matchs consécutifs avec au moins un but (joueurs de champ)
            fld = pp[~pp["is_gk"].astype(bool)].sort_values(["full_name", "season_idx", "round", "date"], kind="mergesort")
            runs = []
            cal = _team_calendar()
            for name, grp in fld.groupby("full_name", sort=False):
                ln, a0, b0 = _longest_scoring_run(grp, cal)
                if ln >= 3:
                    runs.append((ln, name, str(grp["team"].iloc[b0]), str(grp["season"].iloc[a0]), str(grp["season"].iloc[b0]),
                                 int(grp["round"].iloc[a0]), int(grp["round"].iloc[b0]), str(grp["date"].iloc[b0])))
            runs.sort(key=lambda t: (-t[0], t[7]))
            recs.append({"id": "scoring_streak", "label": "Plus longue série de matchs avec un but", "unit": "matchs consécutifs", "top": [
                _row(t[1], t[0], (f"{t[3]} · J{t[5]} → J{t[6]}" if t[3] == t[4] else f"{t[3]} J{t[5]} → {t[4]} J{t[6]}"),
                     team=t[2], season=t[4], date=t[7], kind="player") for t in runs[:N]]})
            cr = pp.groupby("full_name").agg(team=("team", "last"), games=("match_id", "nunique"), goals=("goals", "sum"),
                                              ast=("ast", "sum"), saves=("saves", "sum"), g7=("goals_7m", "sum")).reset_index()
            cd = lambda r: f"{int(r.games)} matchs sur la période"
            for col, label, unit in (("goals", "Total de buts", "buts"), ("ast", "Total de passes décisives", "passes"),
                                     ("saves", "Total d'arrêts", "arrêts"), ("games", "Matchs disputés", "matchs")):
                df = cr[cr[col] > 0].sort_values([col, "games"], ascending=[False, True]).head(N)
                recs.append({"id": f"cr_{col}", "label": label, "unit": unit, "top": [
                    _row(r.full_name, getattr(r, col), cd(r), team=r.team, kind="player") for r in df.itertuples()]})
            groups.append({"id": "players_season", "title": "Joueurs : saison, série et cumul",
                           "blurb": "Records sur une saison, séries de buts et totaux sur la période sélectionnée.", "records": recs})

    # Joueur qui n'a plus joué en Starligue depuis une saison antérieure à la dernière : on le signale à côté de son record.
    if pf is not None:
        pr_all = played_rows()
        last_season = pr_all.groupby("full_name")["season"].max()
        latest = str(pr_all["season"].max())
        for g_ in groups:
            for r_ in g_["records"]:
                for row_ in r_["top"]:
                    if row_.get("kind") == "player" and str(last_season.get(row_["holder"], latest)) < latest:
                        row_["gone"] = True
                        row_["detail"] = f"{row_['detail']} · n'est plus en Starligue (dernier match : {last_season.get(row_['holder'])})"

    n_players = int(played_rows()["full_name"].nunique()) if pf is not None else 0
    res = {"period": seasons_raw, "matches": int(len(mp)), "players": n_players, "groups": _jsonable(groups)}
    with _lock:
        _cache[key] = res
    sim.trim_memory()
    return res


def api_records(qs: dict) -> dict:
    return compute_records(qs.get("seasons", ["all"])[0])


def player_records(name: str, is_gk: bool) -> dict:
    """Records personnels sur toute la période couverte par les données (depuis 2007) + places au palmarès de la Starligue."""
    pp = played_rows()
    mine = pp[pp["full_name"] == name]
    if mine.empty:
        return {"best": [], "season_best": [], "placements": [], "streak": None}

    with _lock:
        bests = _cache.setdefault("pl_bests", {})

    def league_best(col: str) -> pd.Series:
        with _lock:
            if col not in bests:
                src = pp if col != "save_ratio" else pp.assign(save_ratio=np.where(pp["shots_faced"] >= 15, 100 * pp["saves"] / pp["shots_faced"].clip(lower=1), np.nan))
                bests[col] = src.groupby("full_name")[col].max()
            return bests[col]

    specs = ([("saves", "Arrêts", 0), ("saves_7m", "Arrêts sur 7 m", 0), ("save_ratio", "% d'arrêts (≥ 15 tirs)", 1), ("match_note", "Note /10", 2)] if is_gk
             else [("goals", "Buts", 0), ("goals_7m", "Buts à 7 m", 0), ("ast", "Passes décisives", 0), ("br", "Balles récupérées", 0), ("tc", "Contres", 0), ("match_note", "Note /10", 2)])
    src_mine = mine.assign(save_ratio=np.where(mine["shots_faced"] >= 15, 100 * mine["saves"] / mine["shots_faced"].clip(lower=1), np.nan))
    best = []
    for col, label, d in specs:
        s = src_mine[src_mine[col].notna()]
        if s.empty or float(s[col].max()) <= 0:
            continue
        r = s.sort_values([col, "date"], ascending=[False, True]).iloc[0]
        lb = league_best(col)
        rank = int((lb > float(r[col]) + 1e-9).sum()) + 1
        best.append({"stat": col, "label": label, "value": float(r[col]), "display": _fmtv(r[col], d),
                     "detail": f"vs {r['opponent']} ({r['score']}) · J{int(r['round'])} · {r['season']}",
                     "rank": rank, "of": int(lb.notna().sum())})

    ps = mine.groupby(["season", "team"]).agg(games=("match_id", "nunique"), goals=("goals", "sum"), ast=("ast", "sum"),
                                              saves=("saves", "sum"), def_act=("br", "sum"), tc=("tc", "sum"), note=("match_note", "mean")).reset_index()
    ps["def_act"] = ps["def_act"] + ps["tc"]
    season_best = []
    cols = ([("saves", "Arrêts sur une saison", 0), ("note", "Meilleure note /10 moyenne", 2)] if is_gk
            else [("goals", "Buts sur une saison", 0), ("ast", "Passes décisives sur une saison", 0), ("def_act", "BR + TC sur une saison", 0), ("note", "Meilleure note /10 moyenne", 2)])
    for col, label, d in cols:
        s = ps[ps["games"] >= (5 if col == "note" else 1)]
        if s.empty or float(s[col].max()) <= 0:
            continue
        r = s.sort_values([col, "games"], ascending=[False, True]).iloc[0]
        season_best.append({"stat": col, "label": label, "display": _fmtv(r[col], d), "detail": f"{r['season']} · {r['team']} · {int(r['games'])} matchs"})

    streak = None
    if not is_gk:
        grp = mine.sort_values(["season_idx", "round", "date"], kind="mergesort")
        ln, a0, b0 = _longest_run((grp["goals"] > 0).to_numpy())
        if ln >= 3:
            streak = {"length": int(ln), "detail": f"{grp['season'].iloc[a0]} J{int(grp['round'].iloc[a0])} → {grp['season'].iloc[b0]} J{int(grp['round'].iloc[b0])}"}

    placements = []
    for grp_ in compute_records("all")["groups"]:
        for rec in grp_["records"]:
            for i, t in enumerate(rec["top"]):
                if t["kind"] == "player" and t["holder"] == name:
                    placements.append({"id": rec["id"], "label": rec["label"], "rank": i + 1, "display": t["display"], "unit": rec["unit"], "detail": t["detail"]})
    placements.sort(key=lambda p: p["rank"])
    return {"best": best, "season_best": season_best, "placements": placements, "streak": streak}


# ---- Fiche joueur & recherche (comparaison de deux joueurs) -----------------
def api_players_search(qs: dict) -> list[dict]:
    """Recherche de joueurs par nom (insensible aux accents) pour alimenter la comparaison."""
    pf = player_frame()
    if pf is None:
        return []
    q = _fold(qs.get("q", [""])[0])
    gk = qs.get("gk", [""])[0]
    exclude = qs.get("exclude", [""])[0]
    seasons_raw = qs.get("seasons", ["all"])[0].strip() or "all"
    pp = _filter_seasons_df(played_rows(), seasons_raw)
    if pp.empty:
        pp = played_rows()
    g = pp.groupby("full_name").agg(team=("team", "last"), games=("match_id", "nunique"), minutes=("est_seconds", "sum"),
                                    note=("match_note", "mean"), gk=("is_gk", "max")).reset_index()
    g = g[g["full_name"] != exclude]
    if gk in ("0", "1"):
        g = g[g["gk"].astype(bool) == (gk == "1")]
    if q:
        g = g[g["full_name"].map(_fold).str.contains(re.escape(q), regex=True)]
    g = g.sort_values(["minutes", "games"], ascending=False).head(int(np.clip(int(qs.get("limit", [12])[0]), 1, 40)))
    return [{"name": r.full_name, "team": str(r.team), "games": int(r.games), "note": round(float(r.note), 2) if pd.notna(r.note) else None,
             "gk": bool(r.gk)} for r in g.itertuples()]


def api_player_detail(qs: dict) -> dict:
    name = qs.get("name", [""])[0].strip()
    seasons_raw = qs.get("seasons", ["all"])[0].strip()
    m, box, seasons = load_data()
    pf = player_frame()
    if not name or pf is None:
        raise ValueError("Joueur introuvable")

    b_all = pf[pf["played_match"]]
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
    tot_min = float(bp["est_seconds"].sum()) / 60.0
    min_m = tot_min / max(games, 1)

    goals, shots = int(bp["goals"].sum()), int(bp["shots"].sum())
    goals_7m, shots_7m = int(bp["goals_7m"].sum()), int(bp["shots_7m"].sum())
    saves, faced = int(bp["saves"].sum()), int(bp["shots_faced"].sum())
    saves_7m, faced_7m = int(bp["saves_7m"].sum()), int(bp["shots_faced_7m"].sum())
    br, tc = int(bp["br"].sum()), int(bp["tc"].sum())
    ast, yc, rc = int(bp["ast"].sum()), int(bp["yc"].sum()), int(bp["rc"].sum())
    ast_min = float(bp.loc[bp["ast_ok"], "est_seconds"].sum()) / 60.0   # minutes sur les saisons où les passes décisives sont relevées
    bp_lost, two_min = int(bp["bp"].sum()), int(bp["2min"].sum())
    note = float(bp["match_note"].mean()) if bp["match_note"].notna().any() else 5.0
    def_note = float(bp["def_note"].mean()) if bp["def_note"].notna().any() else 5.0
    impact_def = float(bp["impact_def"].sum())
    est_clock = bool(bp["has_real_clock"].mean() < 0.5)

    stats = {
        "games": games, "tot_min": round(tot_min, 1), "min_m": round(min_m, 1), "est_clock": est_clock,
        "goals": goals, "shots": shots, "pct": round(100.0 * goals / max(shots, 1), 1) if shots > 0 else 0.0,
        "goals_7m": goals_7m, "shots_7m": shots_7m,
        "pct_7m": round(100.0 * goals_7m / max(shots_7m, 1), 1) if shots_7m > 0 else 0.0,
        "saves": saves, "faced": faced, "save_pct": round(100.0 * saves / max(faced, 1), 1) if faced > 0 else 0.0,
        "saves_7m": saves_7m, "faced_7m": faced_7m,
        "save_pct_7m": round(100.0 * saves_7m / max(faced_7m, 1), 1) if faced_7m > 0 else 0.0,
        "br": br, "tc": tc, "ast": ast, "ast_min": round(ast_min, 1), "yc": yc, "rc": rc, "bp": bp_lost, "two_min": two_min,
        "note": round(note, 2), "note_max": (round(float(bp["match_note"].max()), 2) if bp["match_note"].notna().any() else None),
        "def_note": round(def_note, 2), "impact_def": round(impact_def, 1),
    }

    # Radar normalisé (50 = moyenne de la ligue sur la période pour le même rôle)
    lg = b_per_all[(b_per_all["is_gk"].astype(bool) == is_gk) & (b_per_all["est_seconds"] >= 600)]
    if lg.empty:
        lg = b_per_all[b_per_all["is_gk"].astype(bool) == is_gk]
    lg_min = max(float(lg["est_seconds"].sum()) / 60.0, 1.0)
    p_min = max(tot_min, 1.0)
    lg_note = float(lg["match_note"].mean()) if lg["match_note"].notna().any() else 5.0
    lg_def_note = float(lg["def_note"].mean()) if lg["def_note"].notna().any() else 5.0

    def _norm(val: float, avg: float) -> int:
        if avg <= 0:
            return 50 if val <= 0 else 85
        return int(np.clip(round(50.0 * (val / avg)), 5, 100))

    ax = lambda axis, score, val, avg, key: {"axis": axis, "key": key, "score": score, "val": val, "avg": avg}
    if is_gk:
        lg_sv_pct = 100.0 * float(lg["saves"].sum()) / max(float(lg["shots_faced"].sum()), 1.0)
        lg_7m_pct = 100.0 * float(lg["saves_7m"].sum()) / max(float(lg["shots_faced_7m"].sum()), 1.0)
        lg_sv_60 = 60.0 * float(lg["saves"].sum()) / lg_min
        lg_min_g = float(lg["est_seconds"].mean()) / 60.0
        radar = [
            ax("% Arrêts", _norm(stats["save_pct"], lg_sv_pct), f"{stats['save_pct']} %", f"{lg_sv_pct:.1f} %", "save_pct"),
            ax("% Arrêts 7m", _norm(stats["save_pct_7m"], lg_7m_pct), f"{stats['save_pct_7m']} %", f"{lg_7m_pct:.1f} %", "save_pct_7m"),
            ax("Arrêts / 60m", _norm(60.0 * saves / p_min, lg_sv_60), f"{60.0 * saves / p_min:.1f}", f"{lg_sv_60:.1f}", "saves_60"),
            ax("Note /10", _norm(note, lg_note), f"{note:.2f} / 10", f"{lg_note:.2f} / 10", "note"),
            ax("Temps / match", _norm(min_m, lg_min_g), f"{min_m:.1f} min", f"{lg_min_g:.1f} min", "min_m"),
        ]
    else:
        lg_g_60 = 60.0 * float(lg["goals"].sum()) / lg_min
        lg_sh_pct = 100.0 * float(lg["goals"].sum()) / max(float(lg["shots"].sum()), 1.0)
        lg_ast_min = max(float(lg.loc[lg["ast_ok"], "est_seconds"].sum()) / 60.0, 1.0)
        lg_ast_60 = 60.0 * float(lg["ast"].sum()) / lg_ast_min
        lg_def_60 = 60.0 * float((lg["br"] + lg["tc"]).sum()) / lg_min
        radar = [
            ax("Buts / 60m", _norm(60.0 * goals / p_min, lg_g_60), f"{60.0 * goals / p_min:.1f}", f"{lg_g_60:.1f}", "goals_60"),
            ax("Précision tir", _norm(stats["pct"], lg_sh_pct), f"{stats['pct']} %", f"{lg_sh_pct:.1f} %", "pct"),
            (ax("Passes D. / 60m", _norm(60.0 * ast / ast_min, lg_ast_60), f"{60.0 * ast / ast_min:.1f}", f"{lg_ast_60:.1f}", "ast_60") if ast_min >= 60
             else ax("Passes D. / 60m", 0, "n.d. (relevées depuis 2023/24)", f"{lg_ast_60:.1f}", "ast_60")),
            ax("Défense / 60m", _norm(60.0 * (br + tc) / p_min, lg_def_60), f"{60.0 * (br + tc) / p_min:.1f}", f"{lg_def_60:.1f}", "def_60"),
            ax("Note /10", _norm(note, lg_note), f"{note:.2f} / 10", f"{lg_note:.2f} / 10", "note"),
            ax("Note défensive LNH", _norm(def_note, lg_def_note), f"{def_note:.2f} / 10", f"{lg_def_note:.2f} / 10", "def_note"),
        ]

    # Historique par saison sur l'ensemble des données du joueur
    bp_full = b_all[b_all["full_name"] == name]
    sg = bp_full.groupby(["season", "team"]).agg(
        games=("match_id", "nunique"), seconds=("est_seconds", "sum"), goals=("goals", "sum"), shots=("shots", "sum"),
        saves=("saves", "sum"), faced=("shots_faced", "sum"), note=("match_note", "mean"),
    ).reset_index().sort_values("season")
    seasons_hist = [{
        "season": str(r.season), "team": str(r.team), "games": int(r.games),
        "min_m": round(float(r.seconds) / 60.0 / max(int(r.games), 1), 1),
        "goals": int(r.goals), "pct": round(100.0 * r.goals / max(int(r.shots), 1), 1) if r.shots > 0 else 0.0,
        "saves": int(r.saves), "save_pct": round(100.0 * r.saves / max(int(r.faced), 1), 1) if r.faced > 0 else 0.0,
        "note": round(float(r.note), 2) if pd.notna(r.note) else None,
    } for r in sg.itertuples()]

    # Top 5 matchs individuels sur la période (classés par note LNH)
    bp_m = bp.sort_values(["match_note", "goals", "saves"], ascending=False)
    best_games = [{
        "match_id": str(r.match_id), "date": str(r.date), "season": str(r.season), "team": str(r.team),
        "opponent": str(r.opponent) if pd.notna(r.opponent) else "", "score": str(r.score),
        "goals": int(r.goals), "shots": int(r.shots), "saves": int(r.saves), "faced": int(r.shots_faced),
        "note": round(float(r.match_note), 2) if pd.notna(r.match_note) else None,
    } for r in bp_m.head(5).itertuples()]

    return {
        "name": name, "teams": teams_list, "is_gk": is_gk, "stats": stats, "radar": radar,
        "seasons_history": seasons_hist, "best_games": best_games, "records": player_records(name, is_gk),
    }


# ---- Explorateur de données & Export CSV ------------------------------------
def _prepare_explorer_df(dataset: str) -> pd.DataFrame:
    m, box, _ = load_data()
    if dataset == "players":
        pf = player_frame()
        if pf is None or pf.empty:
            return pd.DataFrame()
        b = pf[pf["played_match"]].copy()
        b["player"] = b["full_name"]
        for col in ["goals", "shots", "saves", "shots_faced", "est_seconds", "ast", "yc", "rc", "br", "bp", "two_min", "match_note"]:
            b[col] = pd.to_numeric(b[col], errors="coerce")
            if col != "match_note":
                b[col] = b[col].fillna(0)
        b["season"] = b["season"].fillna("").astype(str)
        b["opponent"] = b["opponent"].fillna("").astype(str)
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
            "note": "match_note", "seconds_played": "est_seconds",
            "ast": "ast", "yc": "yc", "rc": "rc", "br": "br", "bp": "bp", "2min": "two_min",
            "opponent": "opponent", "date": "date", "season": "season", "round": "round",
        }
        col = sort_map.get(sort_key, "match_note")
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
            "match_id": str(r.match_id),
            "player": r.player,
            "team": r.team,
            "goals_shots": f"{int(r.goals)}/{int(r.shots)}",
            "saves_faced": f"{int(r.saves)}/{int(r.shots_faced)}" if r.shots_faced > 0 else "–",
            "note": round(float(r.match_note), 2) if pd.notna(r.match_note) else "–",
            "minutes": ("" if r.has_real_clock else "~") + f"{int(r.est_seconds) // 60}:{int(r.est_seconds) % 60:02d}",
            "ast": (int(r.ast) if r.ast_ok else "–"),
            "yc": int(r.yc),
            "rc": int(r.rc),
            "br": int(r.br),
            "bp": int(r.bp),
            "2min": int(r.two_min),
            "opponent": r.opponent or "",
            "date": r.date,
            "season": r.season,
            "round": int(r.round),
        } for r in sub.itertuples()]
    else:
        rows = [{
            "match_id": str(r.match_id),
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
def _run_id(season: str, cutoff: int, n: int) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", f"{season}_c{cutoff}_n{n}")


_SIM_THREAD_SLOT = threading.Semaphore(1)
_HELD_LOCKS: list = []
_JOB_CTL: dict[str, dict] = {}      # contrôle des simulations lancées depuis le site : {"cancel": Event, "proc": Popen | None}


class JobCancelled(Exception):
    """Simulation annulée depuis le site."""


class _sim_slot:
    """Une seule simulation à la fois, tous processus confondus (serveur web + générateur de cache) :
    chacune occupe ~220 Mo, et deux en parallèle dépassent un plafond de 512 Mo."""

    def __init__(self, hold_until_exit: bool = False, cancel: threading.Event | None = None):
        self.cancel = cancel
        # hold_until_exit : le créneau n'est rendu qu'à la mort du processus. Sous Linux le noyau libère la mémoire d'un processus
        # AVANT de fermer ses fichiers : la simulation suivante ne démarre donc jamais pendant que celui-ci occupe encore sa mémoire.
        self.hold = hold_until_exit

    def __enter__(self):
        while not _SIM_THREAD_SLOT.acquire(timeout=0.5):      # file d'attente interruptible
            if self.cancel is not None and self.cancel.is_set():
                raise JobCancelled()
        self._f = None
        try:
            import fcntl
            OUT.mkdir(parents=True, exist_ok=True)
            self._f = open(OUT / ".sim.lock", "w")
            while True:
                try:
                    fcntl.flock(self._f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if self.cancel is not None and self.cancel.is_set():
                        self._f.close()
                        _SIM_THREAD_SLOT.release()
                        raise JobCancelled()
                    time.sleep(0.5)
        except JobCancelled:
            raise
        except Exception:   # Windows : verrou de thread seul
            pass
        return self

    def __exit__(self, *a):
        if self.hold:
            _HELD_LOCKS.append(self._f)   # référence conservée : le verrou ne saute pas avant la fin du processus
            _SIM_THREAD_SLOT.release()
            return
        try:
            if self._f:
                self._f.close()
        finally:
            _SIM_THREAD_SLOT.release()


def _wait_for_memory(on_line=None, timeout: float | None = None, cancel: threading.Event | None = None) -> None:
    """Attend que la mémoire libre permette de lancer une simulation ; échoue proprement plutôt que de faire tuer le serveur."""
    timeout = float(os.environ.get("MEM_WAIT_S", 150)) if timeout is None else timeout
    t0 = time.time()
    while True:
        avail = mem_available_mb()
        if avail is None or avail >= MIN_FREE_MB:
            return
        if time.time() - t0 > timeout:
            raise RuntimeError(f"Mémoire insuffisante ({avail:.0f} Mo libres, {MIN_FREE_MB} Mo nécessaires). Réessayez dans un instant.")
        if on_line:
            on_line(f"En attente de mémoire libre ({avail:.0f} Mo disponibles, {MIN_FREE_MB} Mo nécessaires)…")
        sim.trim_memory()
        for _ in range(8):                                    # 4 s, mais l'annulation est prise en compte tout de suite
            if cancel is not None and cancel.is_set():
                raise JobCancelled()
            time.sleep(0.5)


def _simulate(season: str, cutoff: int, n: int, keep_matches: bool = False, on_line=None, inproc: bool = False, slot_held: bool = False,
              ctl: dict | None = None) -> str:
    """Lance simulation_montecarlo.py et renvoie l'identifiant du run.
    Sous-processus par défaut (la mémoire est rendue à la fin) ; inproc=True dans le générateur de cache, qui s'exécute
    déjà dans son propre processus : on évite ainsi de cumuler ses ~140 Mo avec ceux de la simulation."""
    n = int(max(1, min(n, MAX_SIMS)))
    keep_matches = bool(keep_matches and ALLOW_KEEP_MATCHES)
    run_id = _run_id(season, cutoff, n)
    cmd = [sys.executable, str(HERE / "simulation_montecarlo.py"), "--data-dir", str(DATA), "--output-dir", str(OUT / run_id),
           "--season", season, "--cutoff-round", str(cutoff), "--n-simulations", str(n)]
    if not keep_matches:
        cmd.append("--no-export-matches")
    if on_line:
        on_line("En file d'attente : une autre simulation est peut-être en cours…")
    cancel = ctl["cancel"] if ctl else None
    with (contextlib.nullcontext() if slot_held else _sim_slot(cancel=cancel)):
        if cancel is not None and cancel.is_set():
            raise JobCancelled()
        _wait_for_memory(on_line, cancel=cancel)
        if on_line:
            on_line("Simulation en cours…")
        if inproc:
            with contextlib.redirect_stdout(io.StringIO()):
                sim.main(cmd[2:])
            sim.trim_memory()
        else:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=HERE)
            if ctl is not None:
                ctl["proc"] = p
                if cancel.is_set():               # annulée entre-temps
                    p.kill()
            for line in p.stdout:
                if on_line:
                    on_line(line.rstrip())
            rc = p.wait()
            if cancel is not None and cancel.is_set():
                raise JobCancelled()
            if rc != 0:
                raise RuntimeError(f"La simulation {run_id} a échoué")
    (OUT / run_id / "data_fp.txt").write_text(data_fingerprint())
    return run_id


def cancel_job(jid: str) -> dict:
    """Annule une simulation lancée depuis le site : sortie de la file d'attente, ou arrêt du processus en cours (mémoire rendue)."""
    job, ctl = JOBS.get(jid), _JOB_CTL.get(jid)
    if job is None or ctl is None:
        return {"status": "unknown"}
    if job.get("status") in ("done", "error", "cancelled", "finalizing"):
        return {"status": job["status"], "too_late": job["status"] in ("done", "finalizing")}
    ctl["cancel"].set()
    p = ctl.get("proc")
    if p is not None and p.poll() is None:
        try:
            p.terminate()
            threading.Timer(3.0, lambda: p.poll() is None and p.kill()).start()
        except Exception:
            pass
    job["log"].append("Annulation demandée…")
    return {"status": "cancelling"}


def run_job(jid: str, season: str, cutoff: int, n: int, keep_matches: bool):
    job = JOBS[jid]
    ctl = _JOB_CTL.setdefault(jid, {"cancel": threading.Event(), "proc": None})
    n = int(max(1, min(n, MAX_SIMS)))
    job.update(status="queued", run=_run_id(season, cutoff, n), started=time.time())
    run_dir_ = OUT / _run_id(season, cutoff, n)
    existed = run_dir_.is_dir()

    def on_line(line: str):
        job["log"].append(line)
        job["log"] = job["log"][-60:]
        if line.startswith("Simulation en cours"):
            job["status"] = "running"
    try:
        rid = _simulate(season, cutoff, n, keep_matches, on_line, ctl=ctl)
        job["status"] = "finalizing"   # la simulation est finie (créneau libéré) : seule l'écriture des JSON reste à faire
        job["log"].append("Écriture des fichiers…")
        try:   # fichiers statiques du run (affichage instantané + simulateur conditionnel dans le navigateur)
            write_run_bundle(rid)
        except Exception as e:  # noqa
            job["log"].append(f"Bundle JSON non écrit : {e}")
        job["status"] = "done"
    except JobCancelled:
        if not existed:     # sortie partielle d'une simulation interrompue : on ne la laisse pas traîner
            shutil.rmtree(run_dir_, ignore_errors=True)
        job["log"].append("Simulation annulée.")
        job["status"] = "cancelled"
    except Exception as e:  # noqa
        job["log"].append(str(e)); job["status"] = "error"
    finally:
        ctl["proc"] = None
        sim.trim_memory()


def list_runs() -> list[dict]:
    runs = {}
    try:   # runs du cache versionné (présents même sans dossier output/)
        for r in json.loads((CACHE / "stamp.json").read_text(encoding="utf-8")).get("runs", []):
            runs[r["id"]] = {**r, "time": (CACHE / r["file"]).stat().st_mtime if (CACHE / r["file"]).exists() else 0}
    except Exception:
        pass
    for d in OUT.glob("*/summary_storytelling.json"):
        try:
            j = json.loads(d.read_text(encoding="utf-8"))
            rid = d.parent.name
            runs[rid] = {"id": rid, "season": j["season"], "cutoff": j["cutoff_round"], "n": j["n_simulations"], "time": d.stat().st_mtime,
                         "conditional": True}
        except Exception:  # run corrompu ou en cours d'écriture : on l'ignore
            continue
    for r in runs.values():
        if (CACHE / f"{r['id']}.json").exists():
            r["file"] = f"{r['id']}.json"
            r["cond_file"] = f"{r['id']}.cond.json" if (CACHE / f"{r['id']}.cond.json").exists() else None
    return sorted(runs.values(), key=lambda r: -r.get("time", 0))


def read_table(run: Path, stem: str) -> pd.DataFrame:
    pq, gz = run / f"{stem}.parquet", run / f"{stem}.csv.gz"
    return pd.read_parquet(pq) if pq.exists() else pd.read_csv(gz)


def run_dir(rid: str) -> Path:
    d = (OUT / re.sub(r"[^0-9A-Za-z_]", "", rid))
    if not d.is_dir():
        raise FileNotFoundError(rid)
    return d


def _load_run(d: Path) -> dict:
    """Classements de toutes les simulations, sous forme matricielle (S x T), mis en cache (3 runs max)."""
    j_path = d / "summary_storytelling.json"
    key = ("run", d.name, j_path.stat().st_mtime)
    with _lock:
        if key in _cache:
            return _cache[key]
    j = json.loads(j_path.read_text(encoding="utf-8"))
    teams = list(j["teams"])
    ss = read_table(d, "sim_standings")
    ss = ss.assign(Team=ss["Team"].astype(str))
    rk = ss.pivot(index="simulation", columns="Team", values="Rank")[teams]
    pts = ss.pivot(index="simulation", columns="Team", values="Pts")[teams]
    run = {"json": j, "teams": teams, "ss": ss, "sims": rk.index.to_numpy(),
           "rk": rk.to_numpy(dtype=np.int16), "pts": pts.to_numpy(dtype=np.int16)}
    with _lock:
        for k in [k for k in _cache if isinstance(k, tuple) and k[0] == "run"][:-2]:
            _cache.pop(k, None)
        _cache[key] = run
    return run


def _zones(j: dict, ts: list[dict]) -> dict:
    z = j.get("zones")
    if z:
        return z
    top_col = next((c for c in (ts[0] if ts else {}) if re.match(r"^P_Top\d+", c)), "P_Top4")
    return {"podium": 3, "top_k": int(re.sub(r"\D", "", top_col) or 4), "n_relegated": 2}


def api_sim(rid: str) -> dict:
    d = run_dir(rid)
    run = _load_run(d)
    j, ss = run["json"], run["ss"]
    ts = read_table(d, "team_summary").round(4).to_dict("records")
    notable = j["notable_simulations"]
    ids = {"typical": notable["typical_sim"]}
    for t, v in notable["by_team"].items():
        ids[f"best:{t}"], ids[f"worst:{t}"] = v["best_sim"], v["worst_sim"]
    exp = run["pts"].mean(axis=0)
    out_id = int(run["sims"][int(np.argmax(((run["pts"] - exp) ** 2).sum(axis=1)))])
    wanted = set(ids.values()) | {out_id}
    cols = ["Team", "Rank", "Pts", "W", "D", "L", "GF", "GA", "GD"]
    pick = ss[ss["simulation"].isin(wanted)]
    tab = {int(k): g.sort_values("Rank")[cols].to_dict("records") for k, g in pick.groupby("simulation")}
    return _jsonable({
        "season": j["season"], "cutoff": j["cutoff_round"], "n": j["n_simulations"], "teams": j["teams"],
        "summary": ts, "ratings": j.get("ratings_dc", []), "ids": ids, "outlier": out_id, "tables": tab,
        "rank_probs": j["rank_probabilities"], "metrics": j.get("backtest_metrics", {}),
        "rank_metrics": j.get("rank_metrics", {}),
        # forces Dixon & Coles au cutoff + malus d'absence appliqués aux matchs restants
        "dc_meta": j.get("dc_meta", {}),
        "availability": j.get("availability", []),
        "absence_view": j.get("absence_view", {"cutoff": j.get("cutoff_round"), "now_round": None}),
        "model": {k: j.get("params", {}).get(k) for k in
                  ("absence_off_per_share", "absence_def_per_share", "absence_decay", "absence_lookback",
                   "absence_min_games", "absence_core_presence", "gk_weight")},
        "zones": _zones(j, ts),
        "conditional": any((d / f).exists() for f in ("sim_matches.parquet", "sim_matches.csv.gz", "sim_outcomes.npz")),
        "has_matches": any((d / f).exists() for f in ("sim_matches.parquet", "sim_matches.csv.gz")),
    })


# ---- Simulateur conditionnel -----------------------------------------------
def _load_outcomes(d: Path, match_ids: list[str]) -> pd.DataFrame:
    """Issue (1 / N / 2) de chaque simulation pour les matchs demandés.
    Source : sim_matches.parquet si conservé, sinon la version légère sim_outcomes.npz."""
    wanted = [str(x) for x in match_ids]
    cols = ["simulation", "match_id", "hg", "ag"]
    pq, gz, npz = d / "sim_matches.parquet", d / "sim_matches.csv.gz", d / "sim_outcomes.npz"
    if pq.exists():
        try:
            df = pd.read_parquet(pq, columns=cols, filters=[("match_id", "in", wanted)])
        except Exception:  # filtre non supporté pour les colonnes dictionnaire : on filtre côté pandas
            df = pd.read_parquet(pq, columns=cols)
    elif gz.exists():
        df = pd.read_csv(gz, usecols=cols, dtype={"match_id": str})
    elif npz.exists():
        z = np.load(npz, allow_pickle=False)
        ids = z["match_id"].astype(str)
        idx = [int(np.flatnonzero(ids == w)[0]) for w in wanted if (ids == w).any()]
        sims = z["simulation"]
        df = pd.concat([pd.DataFrame({"simulation": sims, "match_id": ids[i], "hg": z["hg"][:, i], "ag": z["ag"][:, i]}) for i in idx],
                       ignore_index=True) if idx else pd.DataFrame(columns=cols)
    else:
        raise FileNotFoundError("sim_matches.parquet / sim_outcomes.npz")
    df["match_id"] = df["match_id"].astype(str)
    df = df[df["match_id"].isin(wanted)]
    res = np.where(df["hg"] > df["ag"], "1", np.where(df["hg"] == df["ag"], "N", "2"))
    wide = pd.DataFrame({"simulation": df["simulation"].to_numpy(), "match_id": df["match_id"].to_numpy(), "res": res})
    return wide.pivot(index="simulation", columns="match_id", values="res")


def _zone_stats(rk: np.ndarray, pts: np.ndarray, z: dict) -> dict:
    T = rk.shape[1]
    return {
        "p1": (rk == 1).mean(axis=0), "podium": (rk <= z["podium"]).mean(axis=0), "top": (rk <= z["top_k"]).mean(axis=0),
        "rel": (rk > T - z["n_relegated"]).mean(axis=0), "mean_rank": rk.mean(axis=0), "exp_pts": pts.mean(axis=0),
    }


def api_conditional(rid: str, body: dict) -> dict:
    d = run_dir(rid)
    locks = body.get("locks") or []
    if not (1 <= len(locks) <= 3):
        raise ValueError("Verrouillez entre 1 et 3 matchs.")
    clean, seen = [], set()
    for l in locks:
        mid, res = str(l.get("match_id", "")), str(l.get("result", "")).upper()
        if res not in ("1", "N", "2"):
            raise ValueError("Issue invalide : choisir 1 (domicile), N (nul) ou 2 (extérieur).")
        if mid in seen:
            raise ValueError("Un même match ne peut être verrouillé qu'une seule fois.")
        seen.add(mid); clean.append((mid, res))
    upcoming = {str(u["match_id"]): u for u in json.loads((d / "upcoming_scores.json").read_text(encoding="utf-8"))}
    for mid, _ in clean:
        if mid not in upcoming:
            raise ValueError(f"Match {mid} introuvable parmi les matchs restants de cette simulation.")

    run = _load_run(d)
    try:
        oc = _load_outcomes(d, [mid for mid, _ in clean])
    except FileNotFoundError:
        raise ValueError("Cette simulation ne contient pas les issues match par match : relancez-la pour activer le simulateur conditionnel.")
    ok = np.ones(len(oc), dtype=bool)
    for mid, res in clean:
        if mid not in oc.columns:
            raise ValueError(f"Match {mid} absent des résultats détaillés.")
        ok &= oc[mid].to_numpy() == res
    pos = np.searchsorted(run["sims"], oc.index.to_numpy()[ok])

    j, teams = run["json"], run["teams"]
    ts = read_table(d, "team_summary").to_dict("records")
    z = _zones(j, ts)
    base = _zone_stats(run["rk"], run["pts"], z)
    n_total, n_match = int(len(run["sims"])), int(len(pos))
    echo = [{"match_id": mid, "result": res, "home": upcoming[mid]["home"], "away": upcoming[mid]["away"], "round": upcoming[mid]["round"],
             "p": float(upcoming[mid]["p1"] if res == "1" else upcoming[mid]["pN"] if res == "N" else upcoming[mid]["p2"])} for mid, res in clean]
    out = {"n_total": n_total, "n_match": n_match, "share": n_match / max(n_total, 1), "locks": echo, "zones": z, "teams": []}
    if n_match == 0:
        out["warning"] = "Aucune simulation ne réunit ces issues : combinaison trop improbable pour ce nombre de simulations."
        return _jsonable(out)
    if n_match < 100:
        out["warning"] = f"Seulement {n_match} simulations respectent ces issues : les écarts sont à prendre avec prudence. Relancez avec davantage de simulations pour affiner."
    rk_c, pts_c = run["rk"][pos], run["pts"][pos]
    cond = _zone_stats(rk_c, pts_c, z)
    T = len(teams)
    rmat = np.stack([(rk_c == r).mean(axis=0) for r in range(1, T + 1)], axis=1)
    for i, t in enumerate(teams):
        out["teams"].append({"team": t,
                             "base": {k: float(v[i]) for k, v in base.items()},
                             "cond": {k: float(v[i]) for k, v in cond.items()},
                             "rank_probs": [float(x) for x in rmat[i]]})
    return _jsonable(out)


# ---- Recherche globale (clubs, joueurs) --------------------------------------
def _search_index() -> dict:
    with _lock:
        if "search_idx" in _cache:
            return _cache["search_idx"]
    m, box, seasons = load_data()
    cur = seasons[-1] if seasons else ""
    long = pd.concat([m[["home_team", "season"]].rename(columns={"home_team": "t"}),
                      m[["away_team", "season"]].rename(columns={"away_team": "t"})]).drop_duplicates()
    clubs = []
    for t, g in long.groupby("t"):
        ss = sorted(g["season"], key=lambda s: seasons.index(s) if s in seasons else -1)
        clubs.append({"team": str(t), "f": _fold(t), "seasons": len(ss), "last_season": ss[-1], "current": cur in ss})
    players = []
    pf = player_frame()
    if pf is not None:
        pp = played_rows().sort_values(["season_idx", "round"], kind="mergesort")
        g = pp.groupby("full_name").agg(team=("team", "last"), season=("season", "last"), games=("match_id", "nunique"),
                                        minutes=("est_seconds", "sum"), note=("match_note", "mean"), gk=("is_gk", "max")).reset_index()
        for r in g.itertuples():
            players.append({"name": str(r.full_name), "f": _fold(r.full_name), "team": str(r.team), "last_season": str(r.season),
                            "games": int(r.games), "minutes": float(r.minutes), "gk": bool(r.gk),
                            "note": round(float(r.note), 2) if pd.notna(r.note) else None})
    idx = {"clubs": clubs, "players": players}
    with _lock:
        _cache["search_idx"] = idx
    return idx


def api_search(qs: dict) -> dict:
    """Recherche insensible aux accents : tous les mots saisis doivent apparaître dans le nom."""
    q = _fold(qs.get("q", [""])[0])
    toks = [t for t in re.split(r"\s+", q) if t]
    if not toks:
        return {"q": q, "clubs": [], "players": []}
    idx = _search_index()

    def rank(f: str) -> int | None:
        if not all(t in f for t in toks):
            return None
        words = f.split()
        if f.startswith(toks[0]):
            return 0
        return 1 if any(w.startswith(toks[0]) for w in words) else 2

    clubs = sorted(((rank(c["f"]), -c["seasons"], c) for c in idx["clubs"] if rank(c["f"]) is not None), key=lambda x: x[:2])
    players = sorted(((rank(p["f"]), -p["minutes"], p) for p in idx["players"] if rank(p["f"]) is not None), key=lambda x: x[:2])
    strip = lambda d: {k: v for k, v in d.items() if k != "f"}
    return {"q": q, "clubs": [strip(c) for _, _, c in clubs[:6]], "players": [strip(p) for _, _, p in players[:8]]}


# ---- Fiche journaliste : J-précédente et J+suivante --------------------------
MILESTONES = {
    "goals": [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500],
    "ast": [50, 100, 150, 200, 250, 300, 400, 500],
    "saves": [250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000],
    "games": [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
}
MILESTONE_WINDOW = {"goals": 10, "ast": 8, "saves": 40, "games": 5}   # « en approche » = à moins de N unités du cap
STAT_LABEL = {"goals": "buts", "ast": "passes décisives", "saves": "arrêts", "games": "matchs"}
ZONE_KEYS = {"p1": "title", "podium": "podium", "top": "top", "rel": "rel"}
PROFILE = [("pts_m", "Points par match", True), ("gf_m", "Buts marqués par match", True), ("ga_m", "Buts encaissés par match", False),
           ("win_pct", "% de victoires", True), ("shoot_pct", "% de réussite aux tirs", True), ("save_pct", "% d’arrêts des gardiens", True),
           ("note", "Note /10 moyenne", True), ("def_note", "Note défensive /10 moyenne", True), ("two_min_m", "Exclusions de 2 min par match", False)]


def _pl(n, one: str, many: str | None = None) -> str:
    n = int(n)
    return f"{n} {one if abs(n) <= 1 else (many or one + 's')}"


def _ord(n: int) -> str:
    return "1re" if n == 1 else f"{n}e"


def _run_fresh(rid: str) -> bool:
    d = OUT / rid
    try:
        return (d / "summary_storytelling.json").exists() and (d / "data_fp.txt").read_text() == data_fingerprint()
    except Exception:
        return False


def _ensure_run(season: str, cutoff: int, n: int, force: bool = False, inproc: bool = False, slot_held: bool = False) -> str:
    n = int(max(1, min(n, MAX_SIMS)))
    rid = _run_id(season, cutoff, n)
    if force or not _run_fresh(rid):
        print(f"[cache] simulation {season} · journée {cutoff} · {n:,} simulations…".replace(",", " "), flush=True)
        t0 = time.time()
        _simulate(season, cutoff, n, inproc=inproc, slot_held=slot_held)
        print(f"[cache]   terminée en {time.time() - t0:.0f} s", flush=True)
    return rid


def _table(mp: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Classement complet (tous les clubs, même sans match joué)."""
    st = standings(mp).set_index("t") if len(mp) else pd.DataFrame(columns=["j", "gf", "ga", "pts", "gd"])
    st = st.reindex(teams).fillna(0)[["j", "gf", "ga", "pts", "gd"]].astype(int).rename_axis("t")
    st = st.sort_values(["pts", "gd", "gf"], ascending=False, kind="mergesort").reset_index()
    st["rank"] = np.arange(1, len(st) + 1)
    return st


def _trail(flags) -> int:
    n = 0
    for f in reversed(list(flags)):
        if not f:
            break
        n += 1
    return n


def _streaks_now(tm: pd.DataFrame, team: str) -> dict:
    t = tm[tm["team"] == team]
    if t.empty:
        return {"win": 0, "unbeaten": 0, "losing": 0, "home_unbeaten": 0}
    h = t[t["home"]]
    return {"win": _trail(t["res"] == "W"), "unbeaten": _trail(t["res"] != "L"), "losing": _trail(t["res"] == "L"),
            "home_unbeaten": _trail(h["res"] != "L") if len(h) else 0}


def _plural_pts(n: int) -> str:
    return _pl(abs(n), "point")


def _stake_lines(rank: int, pts: int, by_rank: list[int], z: dict) -> list[str]:
    """Phrases chiffrées : où se situe le club par rapport aux seuils du classement."""
    T = len(by_rank)
    if max(by_rank, default=0) == 0:
        return ["Début de saison : classement vierge, tous les clubs sont à égalité"]
    lines = []
    if rank == 1:
        lines.append(f"Leader, {_plural_pts(pts - by_rank[1])} d’avance sur le 2e" if T > 1 else "Leader")
    elif pts == by_rank[0]:
        lines.append("À égalité de points avec le leader (derrière à la différence de buts)")
    else:
        lines.append(f"À {_plural_pts(by_rank[0] - pts)} du leader")
    for cut, name in ((z["podium"], "du podium"), (z["top_k"], f"du top {z['top_k']}")):
        if cut <= 1 or cut >= T or (name == f"du top {z['top_k']}" and z["top_k"] == z["podium"]):
            continue
        if rank > cut:
            lines.append(f"À {_plural_pts(by_rank[cut - 1] - pts)} {name}")
        else:
            lines.append(f"{_plural_pts(pts - by_rank[cut])} d’avance sur le {_ord(cut + 1)} ({name.replace('du ', 'hors du ')})")
    safe = T - z["n_relegated"]
    if safe >= 1 and safe < T:
        if rank > safe:
            lines.append(f"Zone de relégation : à {_plural_pts(by_rank[safe - 1] - pts)} du maintien")
        else:
            lines.append(f"{_plural_pts(pts - by_rank[safe])} d’avance sur la zone de relégation")
    return lines


def _rank_after(table: pd.DataFrame, home: str, away: str, outcome: str) -> dict:
    t = table.set_index("t")[["pts", "gd", "gf"]].copy()
    if outcome == "1":
        t.loc[home, ["pts", "gd"]] += [2, 1]; t.loc[away, "gd"] -= 1
    elif outcome == "2":
        t.loc[away, ["pts", "gd"]] += [2, 1]; t.loc[home, "gd"] -= 1
    else:
        t.loc[[home, away], "pts"] += 1
    t = t.sort_values(["pts", "gd", "gf"], ascending=False, kind="mergesort")
    pos = {name: i + 1 for i, name in enumerate(t.index)}
    return {home: pos[home], away: pos[away]}


def _form(tm: pd.DataFrame, team: str, n: int = 5) -> list[dict]:
    t = tm[tm["team"] == team].tail(n)
    return [{"res": r.res, "opp": r.opp, "home": bool(r.home), "score": f"{int(r.gf)}–{int(r.ga)}", "round": int(r.round), "match_id": str(r.match_id)} for r in t.itertuples()]


def _h2h(m_played: pd.DataFrame, a: str, b: str, exclude_mid: str | None = None) -> dict:
    d = m_played[((m_played["home_team"] == a) & (m_played["away_team"] == b)) | ((m_played["home_team"] == b) & (m_played["away_team"] == a))]
    if exclude_mid:
        d = d[d["match_id"] != exclude_mid]
    d = d.sort_values(["season_idx", "round"], kind="mergesort")
    if d.empty:
        return {"n": 0}
    rows = []
    for r in d.itertuples():
        ga, gb = (r.home_ft_goals, r.away_ft_goals) if r.home_team == a else (r.away_ft_goals, r.home_ft_goals)
        rows.append({"match_id": str(r.match_id), "date": _iso_date(r.date), "season": str(r.season), "round": int(r.round), "home": r.home_team, "away": r.away_team,
                     "score": f"{int(r.home_ft_goals)}–{int(r.away_ft_goals)}", "a_goals": int(ga), "b_goals": int(gb),
                     "winner": a if ga > gb else b if gb > ga else None, "a_home": r.home_team == a})
    wa = sum(1 for x in rows if x["winner"] == a); wb = sum(1 for x in rows if x["winner"] == b); dr = len(rows) - wa - wb
    big = max(rows, key=lambda x: abs(x["a_goals"] - x["b_goals"]))
    streak, who = 0, None
    for x in reversed(rows):
        if x["winner"] is None:
            break
        if who is None:
            who = x["winner"]
        if x["winner"] != who:
            break
        streak += 1
    at_home = [x for x in rows if x["a_home"]]
    return {"n": len(rows), "a": a, "b": b, "w_a": wa, "d": dr, "w_b": wb,
            "gf_a": sum(x["a_goals"] for x in rows), "gf_b": sum(x["b_goals"] for x in rows),
            "avg_total": round(sum(x["a_goals"] + x["b_goals"] for x in rows) / len(rows), 1),
            "last": list(reversed(rows[-5:])),
            "biggest": {"winner": (a if big["a_goals"] > big["b_goals"] else b if big["b_goals"] > big["a_goals"] else None),
                        "score": f"{max(big['a_goals'], big['b_goals'])}–{min(big['a_goals'], big['b_goals'])}", "date": big["date"], "season": big["season"]},
            "streak": {"team": who, "n": streak} if who and streak >= 2 else None,
            "a_home": {"n": len(at_home), "w": sum(1 for x in at_home if x["winner"] == a), "d": sum(1 for x in at_home if x["winner"] is None)}}


def _build_swings(rid: str, mids: list[str], teams_by_mid: dict) -> dict:
    """Pour chaque match : probabilités de zone (titre, podium, top N, relégation) de ses deux clubs, avant et
    selon chacune des 3 issues, à partir des simulations de la run `rid`."""
    d = OUT / rid
    run = _load_run(d)
    ts = read_table(d, "team_summary").to_dict("records")
    z = _zones(run["json"], ts)
    base = _zone_stats(run["rk"], run["pts"], z)
    oc = _load_outcomes(d, mids)
    sims = oc.index.to_numpy()
    pos_all = np.searchsorted(run["sims"], sims)
    out = {}
    for mid in mids:
        if mid not in oc.columns:
            continue
        codes = oc[mid].to_numpy()
        ts2 = teams_by_mid[mid]
        idx = {t: run["teams"].index(t) for t in ts2}
        get = lambda st, t: {k: float(v[idx[t]]) for k, v in st.items()}
        res = {"share": {}, "base": {t: get(base, t) for t in ts2}, "cond": {}}
        for code in ("1", "N", "2"):
            pos = pos_all[codes == code]
            res["share"][code] = float(len(pos) / max(len(sims), 1))
            if len(pos) >= 30:
                cs = _zone_stats(run["rk"][pos], run["pts"][pos], z)
                res["cond"][code] = {t: get(cs, t) for t in ts2}
            else:
                res["cond"][code] = None
        out[mid] = res
    return {"zones": z, "n_sims": int(len(run["sims"])), "swings": out}


def _swing_stake(sw: dict, home: str, away: str) -> dict:
    """Plus grand écart de probabilité (en points) entre une victoire et une défaite d'un des deux clubs."""
    best = {"score": 0.0, "zone": None, "team": None}
    for team, win, loss in ((home, "1", "2"), (away, "2", "1")):
        cw, cl = sw["cond"].get(win), sw["cond"].get(loss)
        if not cw or not cl:
            continue
        for k, zone in ZONE_KEYS.items():
            diff = abs(cw[team][k] - cl[team][k]) * 100
            if diff > best["score"]:
                best = {"score": round(diff, 1), "zone": zone, "team": team}
    return best


def build_press(n: int = CACHE_SIMS) -> dict | None:
    """Fiches journaliste de la dernière journée jouée (J-précédente) et de la suivante (J+suivante)."""
    m, box, seasons = load_data()
    if not seasons:
        return None
    S = seasons[-1]
    mS = m[m["season"] == S].copy()
    played = mS[mS["played"]]
    K = int(played["round"].max()) if len(played) else 0
    next_round = K + 1 if ((mS["round"] == K + 1) & ~mS["played"]).any() else None
    prev_round = K if K >= 1 else None
    if next_round is None and prev_round is None:
        return None
    teams = sorted(set(mS["home_team"]) | set(mS["away_team"]))
    rid_next = _run_id(S, K, min(n, MAX_SIMS)) if next_round else None
    rid_prev = _run_id(S, K - 1, min(n, MAX_SIMS)) if prev_round else None
    for rid_ in (rid_next, rid_prev):
        if rid_ and not _run_fresh(rid_):
            raise RuntimeError(f"simulation {rid_} manquante ou périmée")

    pf = player_frame()
    pfp = played_rows() if pf is not None else pd.DataFrame()
    s_idx = int(mS["season_idx"].iloc[0])
    SINCE = _since()
    SINCE_AST = f"depuis {DP_RELIABLE_FROM[:4]}"     # les passes décisives ne sont relevées qu'à partir de 2023/24
    since_for = lambda stat: SINCE_AST if stat == "ast" else SINCE
    cal = _team_calendar()
    cur_key = s_idx * 1000 + K
    team_last = {}      # club -> rang de son dernier match joué (calendrier du club)
    for (t_, _mid), (ti_, _f, _l, si_, r_) in cal.items():
        if si_ * 1000 + r_ <= cur_key and ti_ > team_last.get(t_, -1):
            team_last[t_] = ti_
    if len(pfp):
        pfp = pfp.assign(key=pfp["season_idx"].astype(np.int32) * 1000 + pfp["round"].astype(np.int32))
    gk_of = pfp.groupby("full_name")["is_gk"].max().to_dict() if len(pfp) else {}
    rec = {r["id"]: r for g in compute_records("all")["groups"] for r in g["records"]}
    tmetrics = {t["team"]: t for t in api_teams({"seasons": [S]})["teams"]}
    mall = m[m["played"]].copy()

    def tot(upto_key: int | None, below: bool = False) -> pd.DataFrame:
        d = pfp if upto_key is None else pfp[pfp["key"] < upto_key] if below else pfp[pfp["key"] <= upto_key]
        if d.empty:
            return pd.DataFrame(columns=["goals", "ast", "saves", "games"])
        g = d.groupby("full_name").agg(goals=("goals", "sum"), ast=("ast", "sum"), saves=("saves", "sum"), games=("match_id", "size"))
        return g

    # joueurs de la saison (duels)
    ps = (pfp[pfp["season"] == S].groupby(["full_name", "team"]).agg(games=("match_id", "nunique"), minutes=("est_seconds", "sum"), goals=("goals", "sum"),
          shots=("shots", "sum"), ast=("ast", "sum"), saves=("saves", "sum"), faced=("shots_faced", "sum"), note=("match_note", "mean"), gk=("is_gk", "max")).reset_index()
          if len(pfp) else pd.DataFrame())

    def duels(team: str, team_games: int) -> dict:
        d = ps[ps["team"] == team] if len(ps) else ps
        if d is None or d.empty:
            return {}
        out = {}
        f = d[~d["gk"].astype(bool)]
        if len(f) and f["goals"].max() > 0:
            r = f.sort_values(["goals", "games"], ascending=[False, True]).iloc[0]
            out["scorer"] = {"name": r.full_name, "goals": int(r.goals), "shots": int(r.shots), "games": int(r.games)}
        if len(f) and f["ast"].max() > 0:
            r = f.sort_values(["ast", "games"], ascending=[False, True]).iloc[0]
            out["playmaker"] = {"name": r.full_name, "ast": int(r.ast), "games": int(r.games)}
        g = d[d["gk"].astype(bool)]
        if len(g):
            r = g.sort_values("minutes", ascending=False).iloc[0]
            out["keeper"] = {"name": r.full_name, "saves": int(r.saves), "faced": int(r.faced), "pct": round(100 * r.saves / max(r.faced, 1), 1),
                             "note": round(float(r.note), 2) if pd.notna(r.note) else None}
        st_ = d[d["games"] >= max(3, int(0.4 * team_games))]
        if len(st_) and st_["note"].notna().any():
            r = st_.sort_values("note", ascending=False).iloc[0]
            out["star"] = {"name": r.full_name, "note": round(float(r.note), 2), "games": int(r.games)}
        return out

    def profile(home: str, away: str) -> list[dict]:
        rows = []
        for key, label, hi in PROFILE:
            vals = {t: v[key] for t, v in tmetrics.items() if v.get(key) is not None and pd.notna(v.get(key))}
            if home not in vals or away not in vals:
                continue
            order = sorted(vals, key=lambda t: -vals[t] if hi else vals[t])
            rows.append({"key": key, "label": label, "higher_better": hi, "home": round(float(vals[home]), 2), "away": round(float(vals[away]), 2),
                         "rank_home": order.index(home) + 1, "rank_away": order.index(away) + 1, "avg": round(float(np.mean(list(vals.values()))), 2), "of": len(vals)})
        return rows

    def milestones_next(team: str, ids: list[str], totals: pd.DataFrame) -> list[dict]:
        names = set(pfp[(pfp["match_id"].isin(ids)) & (pfp["team"] == team)]["full_name"]) if len(pfp) else set()
        items = []
        for nm in names:
            if nm not in totals.index:
                continue
            r = totals.loc[nm]
            stats = ("saves", "games") if gk_of.get(nm) else ("goals", "ast", "games")
            for stat in stats:
                val = int(r[stat]); nxt = next((t for t in MILESTONES[stat] if t > val), None)
                if nxt and nxt - val <= MILESTONE_WINDOW[stat]:
                    items.append({"name": nm, "team": team, "stat": stat, "total": val, "target": nxt, "remaining": nxt - val,
                                  "text": f"{nm} ({team}) : {val} {STAT_LABEL[stat]} en Starligue {since_for(stat)}, plus que {nxt - val} pour atteindre {nxt}"})
        items.sort(key=lambda x: (x["remaining"] / MILESTONE_WINDOW[x["stat"]], -x["total"]))
        return items[:5]

    def milestones_crossed(names_by_team: dict, before: pd.DataFrame, after: pd.DataFrame) -> list[dict]:
        items = []
        for team, names in names_by_team.items():
            for nm in names:
                if nm not in after.index:
                    continue
                for stat in (("saves", "games") if gk_of.get(nm) else ("goals", "ast", "games")):
                    a_ = int(after.loc[nm, stat]); b_ = int(before.loc[nm, stat]) if nm in before.index else 0
                    for thr in MILESTONES[stat]:
                        if b_ < thr <= a_:
                            items.append({"name": nm, "team": team, "stat": stat, "target": thr, "total": a_,
                                          "text": f"{nm} ({team}) passe la barre des {thr} {STAT_LABEL[stat]} en Starligue {since_for(stat)} ({a_} au total)"})
        return items[:8]

    def records_next(team: str, tm_state: pd.DataFrame, ids: list[str], team_played: int, team_total: int, table_row) -> list[dict]:
        items = []
        cs = _streaks_now(tm_state, team)
        for key, rid_, noun, nxt_txt in (("win", "win_streak", "victoires de suite", "Une victoire en ferait {n}"),
                                         ("unbeaten", "unbeaten", "matchs sans défaite", "Un match sans défaite en ferait {n}"),
                                         ("home_unbeaten", "home_unbeaten", "matchs invaincu à domicile", None),
                                         ("losing", "loss_streak", "défaites de suite", None)):
            cur = cs[key]
            r0 = rec.get(rid_, {}).get("top", [None])[0] if rec.get(rid_, {}).get("top") else None
            if cur >= (3 if key != "unbeaten" else 5) and r0:
                txt = f"{team} reste sur {cur} {noun} (record en Starligue {SINCE} : {int(r0['value'])}, {r0['holder']}, {r0['season']})"
                if nxt_txt and key != "home_unbeaten":
                    txt += f". {nxt_txt.format(n=cur + 1)}"
                items.append({"kind": "streak", "team": team, "text": txt})
        # séries de buts individuelles
        if len(pfp):
            r0 = rec.get("scoring_streak", {}).get("top", [None])
            r0 = r0[0] if r0 else None
            sq = pfp[(pfp["match_id"].isin(ids)) & (pfp["team"] == team) & (~pfp["is_gk"].astype(bool))]["full_name"].unique()
            runs_ = []
            for nm in sq:
                g = pfp[(pfp["full_name"] == nm) & (pfp["key"] <= s_idx * 1000 + K)].sort_values("key")
                run_ = _current_scoring_run(g, cal, team, team_last.get(team, -1))
                if run_ >= 4 and r0:
                    runs_.append((run_, nm))
            for run_, nm in sorted(runs_, reverse=True)[:2]:   # les 2 plus longues séries du club
                items.append({"kind": "player_streak", "team": team, "text": f"{nm} ({team}) a marqué lors de ses {run_} derniers matchs (record en Starligue {SINCE} : {int(r0['value'])}, {r0['holder']}{" – n'est plus en Starligue" if r0.get('gone') else ''})"})
            # rythme de saison vs record
            for col, rid_, noun in (("goals", "ps_goals", "buts"), ("ast", "ps_ast", "passes décisives"), ("saves", "ps_saves", "arrêts")):
                recs = rec.get(rid_, {}).get("top")
                if not recs or not team_played:
                    continue
                d = ps[(ps["team"] == team)] if len(ps) else ps
                if d.empty:
                    continue
                d = d[d["gk"].astype(bool)] if col == "saves" else d[~d["gk"].astype(bool)]
                if d.empty or d[col].max() <= 0:
                    continue
                r = d.sort_values(col, ascending=False).iloc[0]
                proj = float(r[col]) / team_played * team_total
                top = recs[0]
                if proj >= 0.9 * top["value"] and int(r[col]) >= 0.4 * top["value"]:
                    if top["holder"] == r.full_name and top["season"] == S:
                        txt = f"{r.full_name} ({team}) détient actuellement le record en Starligue {since_for(col)} de {noun} sur une saison ({int(r[col])})"
                    else:
                        txt = f"{r.full_name} ({team}) : {int(r[col])} {noun} cette saison, {int(round(proj))} à ce rythme (record en Starligue {since_for(col)} : {int(top['value'])}, {top['holder']}, {top['season']})"
                    items.append({"kind": "season_pace", "team": team, "text": txt})
        # rythme du club
        if team_played >= 5:
            for rid_, val, noun, better_low in (("ts_pts_max", table_row.pts, "points", False), ("ts_gf_max", table_row.gf, "buts marqués", False), ("ts_ga_min", table_row.ga, "buts encaissés", True)):
                recs = rec.get(rid_, {}).get("top")
                if not recs:
                    continue
                proj = val / team_played * team_total; top = recs[0]
                if (proj <= 1.05 * top["value"]) if better_low else (proj >= 0.95 * top["value"]):
                    items.append({"kind": "team_pace", "team": team, "text": f"{team} : {int(val)} {noun} en {team_played} matchs, {int(round(proj))} à ce rythme sur la saison (record en Starligue {SINCE} : {int(top['value'])}, {top['holder']}, {top['season']})"})
        return items

    def records_made(mrow, names_by_team: dict, tm_after: pd.DataFrame) -> list[dict]:
        """Records en Starligue établis ou égalés lors de ce match (classements « top 5 » datés du jour du match)."""
        items = []
        d0 = str(mrow.date)
        h, a = mrow.home_team, mrow.away_team
        for rid_, r in rec.items():
            for i, t in enumerate(r["top"]):
                if t["date"] != d0:
                    continue
                ok_ = (t["kind"] == "team" and t["holder"] in (h, a)) or (t["kind"] == "match" and t["holder"] == f"{h} – {a}") or \
                      (t["kind"] == "player" and t["team"] in (h, a) and t["holder"] in set().union(*names_by_team.values()))
                if ok_:
                    items.append({"kind": "record", "rank": i + 1, "text": f"{'RECORD EN STARLIGUE' if i == 0 else 'N°' + str(i + 1) + ' en Starligue'} ({since_for('ast') if str(rid_).endswith('_ast') else SINCE}) · {r['label']} : {t['holder']}, {t['display']} {r['unit']} ({t['detail']})"})
        items.sort(key=lambda x: x["rank"])
        for team in (h, a):
            cs = _streaks_now(tm_after, team)
            if cs["win"] >= 4:
                r0 = rec.get("win_streak", {}).get("top", [None]); r0 = r0[0] if r0 else None
                items.append({"kind": "streak", "rank": 99, "text": f"{team} signe sa {_ord(cs['win'])} victoire de suite" + (f" (record en Starligue {SINCE} : {int(r0['value'])})" if r0 else "")})
        return items[:8]

    def performers(mid: str, team: str) -> dict:
        d = pfp[(pfp["match_id"] == mid) & (pfp["team"] == team)] if len(pfp) else pfp
        out = {}
        if d is None or d.empty:
            return out
        r = d.sort_values(["match_note", "goals"], ascending=False).iloc[0]
        out["best"] = {"name": r.full_name, "note": round(float(r.match_note), 2) if pd.notna(r.match_note) else None, "goals": int(r.goals), "shots": int(r.shots), "saves": int(r.saves), "gk": bool(r.is_gk)}
        f = d[~d["is_gk"].astype(bool)]
        if len(f) and f["goals"].max() > 0:
            r = f.sort_values("goals", ascending=False).iloc[0]
            out["scorer"] = {"name": r.full_name, "goals": int(r.goals), "shots": int(r.shots)}
        g = d[d["is_gk"].astype(bool)]
        if len(g):
            r = g.sort_values("shots_faced", ascending=False).iloc[0]
            out["keeper"] = {"name": r.full_name, "saves": int(r.saves), "faced": int(r.shots_faced), "pct": round(100 * r.saves / max(r.shots_faced, 1), 1)}
        return out

    def make_round(rnd: int, mode: str) -> dict:
        rid = rid_next if mode == "next" else rid_prev
        before_upto = K if mode == "next" else K - 1
        mp_b = mS[mS["played"] & (mS["round"] <= before_upto)]
        mp_a = mS[mS["played"] & (mS["round"] <= K)] if mode == "prev" else None
        tb = _team_match_table(mp_b) if len(mp_b) else pd.DataFrame(columns=["team", "res", "home", "opp", "gf", "ga", "round"])
        ta = _team_match_table(mp_a) if mp_a is not None and len(mp_a) else tb
        tab_b = _table(mp_b, teams)
        tab_a = _table(mp_a, teams) if mp_a is not None else None
        rows = mS[mS["round"] == rnd].sort_values(["date", "match_id"])
        mids = [str(x) for x in rows["match_id"]]
        sw_pack = _build_swings(rid, mids, {str(r.match_id): (r.home_team, r.away_team) for r in rows.itertuples()})
        z = sw_pack["zones"]
        up = {str(u["match_id"]): u for u in json.loads((OUT / rid / "upcoming_scores.json").read_text(encoding="utf-8"))}
        cur_key = prev_key = s_idx * 1000 + K
        sheets = []
        for r in rows.itertuples():
            mid, h, a = str(r.match_id), r.home_team, r.away_team
            sw = sw_pack["swings"].get(mid)
            u = up.get(mid, {})
            by_rank = tab_b["pts"].tolist()
            tinfo = {}
            for t in (h, a):
                row = tab_b[tab_b["t"] == t].iloc[0]
                tinfo[t] = {"rank": int(row["rank"]), "pts": int(row["pts"]), "j": int(row["j"]), "gd": int(row["gd"]), "gf": int(row["gf"]), "ga": int(row["ga"]),
                            "lines": _stake_lines(int(row["rank"]), int(row["pts"]), by_rank, z)}
                if tab_a is not None:
                    ra = tab_a[tab_a["t"] == t].iloc[0]
                    tinfo[t]["after"] = {"rank": int(ra["rank"]), "pts": int(ra["pts"]), "j": int(ra["j"]), "gd": int(ra["gd"])}
            stake = _swing_stake(sw, h, a) if sw else {"score": 0.0, "zone": None, "team": None}
            sheet = {"match_id": mid, "round": int(r.round), "date": _iso_date(r.date), "time": (str(r.time)[:5] if "time" in rows.columns and pd.notna(r.time) else ""),
                     "venue": (str(r.venue) if "venue" in rows.columns and pd.notna(r.venue) and str(r.venue).strip() else ""),
                     "home": h, "away": a, "status": "played" if mode == "prev" else "upcoming", "teams": tinfo,
                     "form": {h: _form(tb, h), a: _form(tb, a)},
                     "proba": {k: u.get(k) for k in ("p1", "pN", "p2", "exp_h", "exp_a", "modal")},
                     "swing": sw, "stake": stake,
                     "profile": profile(h, a), "duels": {h: duels(h, int(tinfo[h]["j"])), a: duels(a, int(tinfo[a]["j"]))},
                     "h2h": _h2h(mall, h, a, exclude_mid=mid if mode == "prev" else None)}
            if mode == "next":
                sheet["rank_after"] = {c: _rank_after(tab_b, h, a, c) for c in ("1", "N", "2")}
                last5 = lambda t: [str(x) for x in mS[(mS["played"]) & (mS["round"] <= K) & ((mS["home_team"] == t) | (mS["away_team"] == t))].sort_values("round")["match_id"].tail(5)]
                totals = tot(cur_key)
                tp = lambda t: int(tab_b[tab_b["t"] == t].iloc[0]["j"])
                tt = lambda t: int(((mS["home_team"] == t) | (mS["away_team"] == t)).sum())
                row_of = lambda t: tab_b[tab_b["t"] == t].iloc[0]
                sheet["milestones"] = milestones_next(h, last5(h), totals) + milestones_next(a, last5(a), totals)
                sheet["records"] = records_next(h, tb, last5(h), tp(h), tt(h), row_of(h)) + records_next(a, tb, last5(a), tp(a), tt(a), row_of(a))
            else:
                hg, ag = int(r.home_ft_goals), int(r.away_ft_goals)
                code = "1" if hg > ag else "2" if ag > hg else "N"
                ht = None
                if "home_ht_goals" in rows.columns and pd.notna(r.home_ht_goals) and pd.notna(r.away_ht_goals):
                    ht = [int(r.home_ht_goals), int(r.away_ht_goals)]
                p_real = {"1": u.get("p1"), "N": u.get("pN"), "2": u.get("p2")}.get(code)
                sheet["result"] = {"hg": hg, "ag": ag, "ht": ht, "code": code, "p_real": p_real,
                                   "surprise": bool(p_real is not None and p_real < 0.25)}
                sheet["performers"] = {h: performers(mid, h), a: performers(mid, a)}
                names = {t: set(pfp[(pfp["match_id"] == mid) & (pfp["team"] == t)]["full_name"]) if len(pfp) else set() for t in (h, a)}
                before_t, after_t = tot(prev_key, below=True), tot(prev_key)
                sheet["milestones"] = milestones_crossed(names, before_t, after_t)
                sheet["records"] = records_made(r, names, ta)
            sheets.append(sheet)
        # affiche de la journée = plus fort enjeu chiffré
        order = sorted(range(len(sheets)), key=lambda i: -sheets[i]["stake"]["score"])
        for pos_, i in enumerate(order):
            sheets[i]["stake_rank"] = pos_ + 1
        dates = sorted(x["date"] for x in sheets if x["date"])
        return {"round": rnd, "run": rid, "dates": [dates[0], dates[-1]] if dates else [], "zones": z, "n_sims": sw_pack["n_sims"], "matches": sheets}

    out = {"version": 1, "season": S, "since": SINCE, "teams": teams, "last_played_round": K, "built": time.strftime("%Y-%m-%d %H:%M:%S"), "data_stamp": _stamp(),
           "next": make_round(next_round, "next") if next_round else None,
           "prev": make_round(prev_round, "prev") if prev_round else None}
    return out


# ---- Fiche d'un match passé (modale) -------------------------------------------
def api_match(qs: dict) -> dict:
    """Résumé « façon fiche journaliste » d'un match déjà joué + feuille de match.
    Pas de probabilités (elles viennent de simulations lancées pour la saison en cours seulement) : classement avant/après,
    forme, face-à-face, caps franchis, records établis et statistiques de chaque joueur."""
    mid = str((qs.get("id") or [""])[0]).strip()
    with _lock:
        mc = _cache.setdefault("match_sheets", {})
        if mid in mc:
            return mc[mid]
    m, box, seasons = load_data()
    row = m[m["match_id"].astype(str) == mid]
    if row.empty:
        raise ValueError("Match introuvable")
    r = row.iloc[0]
    if not bool(r["played"]):
        raise ValueError("Ce match n'a pas encore été joué")
    S, rnd, h, a = str(r["season"]), int(r["round"]), str(r["home_team"]), str(r["away_team"])
    d0 = _iso_date(r["date"])
    mS = m[m["season"] == S]
    teams = sorted(set(mS["home_team"]) | set(mS["away_team"]))
    z = {"title": 1, "podium": 3, "top_k": 4, "n_relegated": 2}
    SINCE = _since()

    def before_mask(df: pd.DataFrame) -> pd.Series:
        dd = df["date"].map(_iso_date)
        if d0:
            return df["played"] & (((dd != "") & (dd < d0)) | ((dd == d0) & (df["round"] < rnd)))
        return df["played"] & (df["round"] < rnd)

    this = mS[mS["match_id"].astype(str) == mid]
    bS = mS[before_mask(mS) & (mS["match_id"].astype(str) != mid)]
    aS = pd.concat([bS, this])
    tab_b, tab_a = _table(bS, teams), _table(aS, teams)
    by_rank = tab_b["pts"].tolist()
    tinfo = {}
    for t in (h, a):
        rb, ra = tab_b[tab_b["t"] == t].iloc[0], tab_a[tab_a["t"] == t].iloc[0]
        tinfo[t] = {"rank": int(rb["rank"]), "pts": int(rb["pts"]), "j": int(rb["j"]), "gd": int(rb["gd"]), "gf": int(rb["gf"]), "ga": int(rb["ga"]),
                    "lines": _stake_lines(int(rb["rank"]), int(rb["pts"]), by_rank, z),
                    "after": {"rank": int(ra["rank"]), "pts": int(ra["pts"]), "j": int(ra["j"]), "gd": int(ra["gd"])}}
    tb, ta = _team_match_table(bS) if len(bS) else pd.DataFrame(columns=["team", "res", "home", "opp", "gf", "ga", "round", "match_id"]), _team_match_table(aS)

    allm = m[m["played"]]
    prior = allm[before_mask(allm) & (allm["match_id"].astype(str) != mid)]
    sheet = {"match_id": mid, "season": S, "round": rnd, "date": d0, "time": (str(r["time"])[:5] if "time" in m.columns and pd.notna(r["time"]) else ""),
             "venue": (str(r["venue"]) if "venue" in m.columns and pd.notna(r["venue"]) and str(r["venue"]).strip() else ""),
             "home": h, "away": a, "status": "played", "teams": tinfo,
             "form": {h: _form(tb, h), a: _form(tb, a)}, "profile": [], "duels": {},
             "result": {"hg": int(r["home_ft_goals"]), "ag": int(r["away_ft_goals"]),
                        "ht": ([int(r["home_ht_goals"]), int(r["away_ht_goals"])] if pd.notna(r.get("home_ht_goals")) and pd.notna(r.get("away_ht_goals")) else None)},
             "h2h": _h2h(prior, h, a), "since": SINCE, "milestones": [], "records": [], "performers": {}, "box": {h: [], a: []}}

    # ---- feuille de match + temps forts -----------------------------------------
    pr = played_rows()
    mine = pr[pr["match_id"].astype(str) == mid] if len(pr) else pr
    mine = mine.rename(columns={"2min": "two_min"})
    names_by_team = {t: set(mine[mine["team"] == t]["full_name"]) for t in (h, a)}
    if len(mine):
        for t in (h, a):
            for x in mine[mine["team"] == t].sort_values(["is_gk", "est_seconds"], ascending=[True, False]).itertuples():
                sheet["box"][t].append({
                    "name": str(x.full_name), "gk": bool(x.is_gk), "min": int(x.est_seconds) // 60,
                    "goals": int(x.goals), "shots": int(x.shots), "g7": int(x.goals_7m), "s7": int(x.shots_7m),
                    "saves": int(x.saves), "faced": int(x.shots_faced), "s7_saves": int(x.saves_7m), "s7_faced": int(x.shots_faced_7m),
                    "ast": (int(x.ast) if bool(x.ast_ok) else None), "br": int(x.br), "tc": int(x.tc), "bp": int(x.bp),
                    "yc": int(x.yc), "two": int(x.two_min),
                    "rc": int(x.rc),
                    "note": round(float(x.match_note), 2) if pd.notna(x.match_note) else None})
        for t in (h, a):
            d = mine[mine["team"] == t]
            if d.empty:
                continue
            o = {}
            b = d.sort_values(["match_note", "goals"], ascending=False).iloc[0]
            o["best"] = {"name": b.full_name, "note": round(float(b.match_note), 2) if pd.notna(b.match_note) else None, "goals": int(b.goals), "shots": int(b.shots), "saves": int(b.saves), "gk": bool(b.is_gk)}
            f = d[~d["is_gk"].astype(bool)]
            if len(f) and f["goals"].max() > 0:
                x = f.sort_values("goals", ascending=False).iloc[0]
                o["scorer"] = {"name": x.full_name, "goals": int(x.goals), "shots": int(x.shots)}
            g = d[d["is_gk"].astype(bool)]
            if len(g):
                x = g.sort_values("shots_faced", ascending=False).iloc[0]
                o["keeper"] = {"name": x.full_name, "saves": int(x.saves), "faced": int(x.shots_faced), "pct": round(100 * x.saves / max(x.shots_faced, 1), 1)}
            sheet["performers"][t] = o

        # caps franchis (totaux en Starligue avant / après ce match)
        allnames = set().union(*names_by_team.values())
        pdates = pr["date"].map(_iso_date)
        hist = pr[pr["full_name"].isin(allnames) & (pr["match_id"].astype(str) != mid) & ((pdates < d0) if d0 else (pr["season_idx"] * 1000 + pr["round"] < int(r["season_idx"]) * 1000 + rnd))]
        before_t = hist.groupby("full_name").agg(goals=("goals", "sum"), ast=("ast", "sum"), saves=("saves", "sum"), games=("match_id", "size"))
        for t in (h, a):
            for x in mine[mine["team"] == t].itertuples():
                nm = x.full_name
                b0 = before_t.loc[nm] if nm in before_t.index else None
                gain = {"goals": int(x.goals), "ast": int(x.ast) if bool(x.ast_ok) else 0, "saves": int(x.saves), "games": 1}
                for stat in (("saves", "games") if bool(x.is_gk) else ("goals", "ast", "games")):
                    if stat == "ast" and not bool(x.ast_ok):
                        continue
                    v0 = int(b0[stat]) if b0 is not None else 0
                    v1 = v0 + gain[stat]
                    for thr in MILESTONES[stat]:
                        if v0 < thr <= v1:
                            since = f"depuis {DP_RELIABLE_FROM[:4]}" if stat == "ast" else SINCE
                            sheet["milestones"].append({"name": nm, "team": t, "stat": stat, "target": thr, "total": v1,
                                                        "text": f"{nm} ({t}) passe la barre des {thr} {STAT_LABEL[stat]} en Starligue {since} ({v1} au total)"})

    # ---- records établis ou égalés lors de ce match -----------------------------------
    rec = {x["id"]: x for g_ in compute_records("all")["groups"] for x in g_["records"]}
    allp = set().union(*names_by_team.values()) if names_by_team else set()
    items = []
    for rid_, rr in rec.items():
        for i, t in enumerate(rr["top"]):
            if not d0 or t["date"] != d0:
                continue
            ok_ = (t["kind"] == "team" and t["holder"] in (h, a)) or (t["kind"] == "match" and t["holder"] == f"{h} – {a}") or \
                  (t["kind"] == "player" and t["team"] in (h, a) and t["holder"] in allp)
            if ok_:
                since = f"depuis {DP_RELIABLE_FROM[:4]}" if str(rid_).endswith("_ast") else SINCE
                items.append({"kind": "record", "rank": i + 1, "text": f"{'RECORD EN STARLIGUE' if i == 0 else 'N°' + str(i + 1) + ' en Starligue'} ({since}) · {rr['label']} : {t['holder']}, {t['display']} {rr['unit']} ({t['detail']})"})
    items.sort(key=lambda x: x["rank"])
    for t in (h, a):
        cs = _streaks_now(ta, t)
        if cs["win"] >= 4:
            r0 = (rec.get("win_streak", {}).get("top") or [None])[0]
            items.append({"kind": "streak", "rank": 99, "text": f"{t} signe sa {_ord(cs['win'])} victoire de suite" + (f" (record en Starligue {SINCE} : {int(r0['value'])})" if r0 else "")})
    sheet["records"] = items[:8]

    with _lock:
        mc[mid] = sheet
    return sheet


# ---- Cache JSON : bundles prêts à afficher ---------------------------------
_gz_memo: dict = {}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(obj), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(path)  # écriture atomique : le navigateur ne lit jamais un fichier à moitié écrit


def _b64(a: np.ndarray) -> str:
    import base64
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def build_cond(rid: str) -> dict | None:
    """Données compactes du simulateur conditionnel (« et si… ? ») : le navigateur calcule tout lui-même,
    sans solliciter le serveur. Classements et points de chaque simulation + issue (1/N/2) de chaque match restant
    (2 bits par match)."""
    d = OUT / rid
    npz = d / "sim_outcomes.npz"
    if not npz.exists():
        return None
    run = _load_run(d)
    z_ = np.load(npz, allow_pickle=False)
    if not np.array_equal(z_["simulation"], run["sims"]):
        return None
    ts = read_table(d, "team_summary").to_dict("records")
    code = np.where(z_["hg"] > z_["ag"], 0, np.where(z_["hg"] == z_["ag"], 1, 2)).astype(np.uint8)   # (S, matchs restants)
    flat = code.reshape(-1)
    flat = np.concatenate([flat, np.zeros((-len(flat)) % 4, np.uint8)]).reshape(-1, 4)
    packed = (flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6)).astype(np.uint8)
    S_, T_ = run["rk"].shape
    return {"version": 1, "run": rid, "S": int(S_), "T": int(T_), "Mu": int(code.shape[1]), "teams": run["teams"], "zones": _zones(run["json"], ts),
            "match_ids": [str(x) for x in z_["match_id"]], "ranks": _b64(run["rk"].astype(np.uint8)),
            "pts": _b64(np.clip(run["pts"], 0, 255).astype(np.uint8)), "outcomes": _b64(packed)}


def write_run_bundle(rid: str, runs: list | None = None, meta_: dict | None = None, built: str | None = None) -> dict:
    """Écrit les fichiers statiques d'un run : <rid>.json (tout l'onglet Monte Carlo) et <rid>.cond.json."""
    d = OUT / rid
    b = {"version": 1, "run": rid, "built": built or time.strftime("%Y-%m-%d %H:%M:%S"), "data_fp": data_fingerprint(),
         "sim": api_sim(rid), "upcoming": json.loads((d / "upcoming_scores.json").read_text(encoding="utf-8")), "runs": runs or []}
    cond = build_cond(rid)
    b["sim"]["cond_file"] = f"{rid}.cond.json" if cond else None
    if cond:
        _write_json(CACHE / f"{rid}.cond.json", cond)
    _write_json(CACHE / f"{rid}.json", b)
    return b


def last_played_round(m: pd.DataFrame, season: str) -> int:
    mS = m[(m["season"] == season) & m["played"]]
    return int(mS["round"].max()) if len(mS) else 0


def cache_specs(seasons: list[str], m: pd.DataFrame | None = None) -> list[tuple[str, int]]:
    """Saison en cours (dernière des données) depuis l'avant-dernière journée jouée, puis saison précédente à la J0."""
    if m is None:
        m, _ = load_matches()
    specs = [(seasons[-1], max(last_played_round(m, seasons[-1]) - 1, 0))]
    if len(seasons) >= 2:
        specs.append((seasons[-2], 0))
    return specs


def _cache_status(state: str, message: str = "") -> None:
    try:
        _write_json(CACHE / "status.json", {"state": state, "message": message, "ts": time.time()})
    except Exception:
        pass


def cache_fresh() -> bool:
    """Vrai si les bundles existent et correspondent au contenu actuel des CSV."""
    try:
        st = json.loads((CACHE / "stamp.json").read_text(encoding="utf-8"))
        return (CACHE / "default.json").is_file() and st.get("data_fp") == data_fingerprint()
    except Exception:
        return False


def build_cache(n: int = CACHE_SIMS, force: bool = False) -> list[str]:
    """Génère les simulations puis un bundle JSON par run.
    Deux phases pour borner la mémoire : (1) les simulations, une par une, avec seulement les matchs en mémoire ;
    (2) une fois TOUTES terminées, chargement des feuilles de match pour les bundles et la fiche journaliste."""
    n = int(min(n, MAX_SIMS if MAX_SIMS else n))
    CACHE.mkdir(parents=True, exist_ok=True)
    lock = None
    try:
        import fcntl
        lock = open(CACHE / ".build.lock", "w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[cache] une génération est déjà en cours : abandon.", flush=True)
        return []
    except Exception:
        pass
    try:
        with _sim_slot(hold_until_exit=True):   # une seule chose lourde à la fois : les simulations du site patientent jusqu'à la fin
            _cache_status("building", "Préparation des simulations…")
            m, seasons = load_matches()
            S = seasons[-1]
            mS = m[m["season"] == S]
            K = int(mS[mS["played"]]["round"].max()) if mS["played"].any() else 0
            wanted = list(cache_specs(seasons, m))
            if ((mS["round"] == K + 1) & ~mS["played"]).any():
                wanted.append((S, K))            # fiche journaliste : J+suivante
            if K >= 1:
                wanted.append((S, K - 1))        # fiche journaliste : J-précédente
            wanted = list(dict.fromkeys(wanted))
            # phase 1 : simulations
            for i, (season, cutoff) in enumerate(wanted, 1):
                _cache_status("building", f"Simulation {i}/{len(wanted)} ({season}, journée {cutoff})…")
                _ensure_run(season, cutoff, n, force, inproc=True, slot_held=True)
                sim.trim_memory()
            # phase 2 : bundles + fiche journaliste (toujours sous le créneau : aucune simulation du site en parallèle)
            _cache_status("building", "Écriture des fichiers JSON…")
            load_data()
            fp, built = data_fingerprint(), time.strftime("%Y-%m-%d %H:%M:%S")
            specs = cache_specs(seasons, m)
            runs = [{"id": _run_id(s, c, n), "season": s, "cutoff": c, "n": n, "file": f"{_run_id(s, c, n)}.json", "conditional": True} for s, c in specs]
            rids = [r["id"] for r in runs]
            for r in runs[1:]:
                write_run_bundle(r["id"], runs, built=built)
            first = write_run_bundle(rids[0], runs, built=built)
            try:   # un échec ici ne doit pas empêcher le reste du cache
                press = build_press(n)
                if press:
                    _write_json(CACHE / "press.json", press)
            except Exception as e:  # noqa
                (CACHE / "press.json").unlink(missing_ok=True)
                print(f"[cache] fiche journaliste non générée : {e}", flush=True)
            _write_json(CACHE / "default.json", {**first, "meta": meta()})
            _write_json(CACHE / "stamp.json", {"data_fp": fp, "built": built, "n": n, "runs": runs})
            for f in sorted(CACHE.glob("*.json")):
                print(f"[cache]   {f.name} : {f.stat().st_size / 1024:.0f} Ko", flush=True)
        _cache_status("ready", built)
        return rids
    except Exception as e:  # noqa
        _cache_status("error", str(e))
        raise
    finally:
        sim.trim_memory()
        if lock:
            lock.close()


def spawn_cache_builder() -> None:
    """Lance la génération dans un PROCESSUS À PART : sa mémoire est entièrement rendue à la fin."""
    def preexec():
        try:
            os.nice(10)
        except Exception:
            pass
    try:
        subprocess.Popen([sys.executable, str(HERE / "server.py"), "--build-cache"], cwd=HERE, preexec_fn=preexec if os.name == "posix" else None)
    except Exception as e:  # noqa
        print(f"[cache] impossible de lancer la génération : {e}", flush=True)


def api_cache_status() -> dict:
    """Minuscule (aucune donnée chargée) : permet au navigateur de savoir quand un cache plus récent est prêt."""
    st = {}
    try:
        st = json.loads((CACHE / "status.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    built = None
    try:
        built = json.loads((CACHE / "stamp.json").read_text(encoding="utf-8")).get("built")
    except Exception:
        pass
    fresh = cache_fresh()
    state = "ready" if fresh else st.get("state", "missing")
    if state == "building" and time.time() - st.get("ts", 0) > 3600:
        state = "missing"
    return {"fresh": fresh, "state": state, "built": built if fresh else None, "message": st.get("message", ""), "auto": AUTO_BUILD_CACHE}


_last_trim = [0.0]


def _maybe_trim() -> None:
    """Rend la mémoire inutilisée au système après une requête lourde (au plus toutes les 5 s)."""
    now = time.time()
    if now - _last_trim[0] > 5:
        _last_trim[0] = now
        sim.trim_memory()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, obj, code=200, ctype="application/json", headers: dict | None = None):
        b = obj if isinstance(obj, bytes) else json.dumps(_jsonable(obj), ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        if "Cache-Control" not in (headers or {}):
            self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)

    def send_cached_json(self, name: str):
        """Sert un bundle statique : ETag (304 si inchangé) + gzip."""
        if not re.fullmatch(r"[0-9A-Za-z_]+(\.cond)?\.json", name) or name in ("stamp.json", "status.json"):
            return self.send({"error": "not found"}, 404)
        if name in ("default.json", "press.json") and not cache_fresh():
            return self.send({"error": "cache absent ou périmé"}, 404)
        p = CACHE / name
        if not p.is_file():
            return self.send({"error": "cache absent"}, 404)
        st = p.stat()
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304); self.send_header("ETag", etag); self.send_header("Cache-Control", "no-cache"); self.end_headers()
            return
        hdr = {"ETag": etag, "Cache-Control": "no-cache"}   # no-cache = revalider à chaque fois (réponse 304 quasi instantanée)
        if "gzip" in self.headers.get("Accept-Encoding", ""):
            key = (name, st.st_mtime_ns)
            if key not in _gz_memo:
                _gz_memo.clear()
                _gz_memo[key] = gzip.compress(p.read_bytes(), 6)
            hdr["Content-Encoding"] = "gzip"
            return self.send(_gz_memo[key], headers=hdr)
        return self.send(p.read_bytes(), headers=hdr)

    def do_GET(self):
        u = urlparse(self.path); parts = [x for x in u.path.split("/") if x]
        try:
            if not parts:
                return self.send((HERE / "index.html").read_bytes(), ctype="text/html")
            if parts in (["mentions-legales"], ["legal.html"]):
                return self.send((HERE / "legal.html").read_bytes(), ctype="text/html")
            if len(parts) == 2 and parts[0] == "cache": return self.send_cached_json(parts[1])
            if parts[:2] == ["api", "cache_status"]: return self.send(api_cache_status())
            if parts[:2] == ["api", "meta"]: return self.send(meta())
            if parts[:2] == ["api", "history"]:
                load_data()
                with _lock:
                    if "history_bytes" not in _cache:
                        _cache["history_bytes"] = json.dumps(_jsonable(build_history()), ensure_ascii=False).encode("utf-8")
                        sim.trim_memory()
                    hist = _cache["history_bytes"]
                return self.send(hist)
            if parts[:2] == ["api", "match"]:
                return self.send(api_match(parse_qs(u.query)))
            if parts[:2] == ["api", "club"]:
                return self.send(api_club_detail(parse_qs(u.query)))
            if parts[:2] == ["api", "player"]:
                return self.send(api_player_detail(parse_qs(u.query)))
            if parts[:2] == ["api", "players"]:
                return self.send(api_players_search(parse_qs(u.query)))
            if parts[:2] == ["api", "search"]:
                return self.send(api_search(parse_qs(u.query)))
            if parts[:2] == ["api", "teams"]:
                return self.send(api_teams(parse_qs(u.query)))
            if parts[:2] == ["api", "records"]:
                return self.send(api_records(parse_qs(u.query)))
            if parts[:2] == ["api", "runs"]: return self.send(list_runs())
            if parts[:2] == ["api", "jobs"]: return self.send(JOBS.get(parts[2] if len(parts) > 2 else "", {"status": "unknown"}))
            if parts[:2] == ["api", "sim"] and len(parts) > 2:
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
        except ValueError as e:
            self.send({"error": str(e)}, 400)
        except Exception as e:  # noqa
            self.send({"error": str(e)}, 500)
        finally:
            if parts[:1] == ["api"] and parts[:2] not in (["api", "cache_status"], ["api", "jobs"]):
                _maybe_trim()

    def do_POST(self):
        parts = [x for x in urlparse(self.path).path.split("/") if x]
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except Exception:
            return self.send({"error": "Corps de requête JSON invalide"}, 400)
        if parts[:2] == ["api", "sim"] and len(parts) == 4 and parts[3] == "conditional":
            try:
                return self.send(api_conditional(parts[2], body))
            except FileNotFoundError as e:
                return self.send({"error": f"Fichier introuvable : {e}"}, 404)
            except ValueError as e:
                return self.send({"error": str(e)}, 400)
            except Exception as e:  # noqa
                return self.send({"error": str(e)}, 500)
        if parts[:2] == ["api", "jobs"] and len(parts) == 4 and parts[3] == "cancel":
            return self.send(cancel_job(parts[2]))
        if parts != ["api", "simulate"]:
            return self.send({"error": "not found"}, 404)
        try:
            season, cutoff = str(body["season"]), max(0, int(body.get("cutoff_round", 0)))
            n = int(np.clip(int(body.get("n_simulations", 5000)), 200, max(MAX_SIMS, 200)))
        except Exception:
            return self.send({"error": "Paramètres invalides"}, 400)
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"status": "queued", "log": [], "run": None}
        _JOB_CTL[jid] = {"cancel": threading.Event(), "proc": None}
        threading.Thread(target=run_job, args=(jid, season, cutoff, n, bool(body.get("keep_matches"))), daemon=True).start()
        self.send({"job": jid})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Serveur LNH Starligue")
    ap.add_argument("--build-cache", action="store_true", help="génère les bundles JSON Monte Carlo puis quitte")
    ap.add_argument("--force", action="store_true", help="avec --build-cache : relance aussi les simulations déjà calculées")
    ap.add_argument("--sims", type=int, default=CACHE_SIMS, help=f"simulations par bundle (défaut {CACHE_SIMS})")
    args = ap.parse_args()
    for _stream in (sys.stdout, sys.stderr):   # consoles Windows (cp1252) : ne plante pas sur « → » ou les accents
        try:
            _stream.reconfigure(errors="replace")
        except Exception:
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    if not DATA.is_dir():
        sys.exit(f"Dossier de données introuvable : {DATA}")
    if args.build_cache:
        build_cache(n=args.sims, force=args.force)
        sys.exit(0)
    print(f"Données : {DATA}\nSorties : {OUT}\nCache   : {CACHE}\n→ http://localhost:{PORT}")
    lim = mem_limit_mb()
    print(f"[mémoire] plafond détecté : {'%.0f Mo' % lim if lim else 'aucun'} · une simulation à la fois · {MAX_SIMS} simulations max")
    if cache_fresh():
        print("[cache] bundles JSON à jour : le site s'ouvre directement sur l'onglet Monte Carlo.")
    elif AUTO_BUILD_CACHE:
        print("[cache] absent ou périmé : génération dans un processus séparé (AUTO_BUILD_CACHE=1). Le site fonctionne en attendant.")
        spawn_cache_builder()
    else:
        print("[cache] absent ou périmé : aucune génération automatique (elle consomme ~250 Mo).\n"
              "        → en local : python server.py --build-cache  puis versionnez le dossier ./cache\n"
              "        → ou définissez AUTO_BUILD_CACHE=1 si l'instance a assez de mémoire.")
    srv = ThreadingHTTPServer(("", PORT), H)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
