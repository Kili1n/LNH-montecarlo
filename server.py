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


def _stamp() -> float:
    fs = list(DATA.rglob("*.csv"))
    return max((f.stat().st_mtime for f in fs), default=0.0)


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


def top_players(box: pd.DataFrame | None, season: str | None) -> dict:
    if box is None:
        return {"mvp": [], "scorers": [], "keepers": []}
    b = box if season is None else box[box["season"] == season]
    b = b.assign(name=(b["first_name"].fillna("") + " " + b["last_name"].fillna("")).str.strip())
    g = b.groupby(["name", "team"]).agg(
        games=("match_id", "nunique"), goals=("goals", "sum"), shots=("shots", "sum"), saves=("saves", "sum"),
        faced=("shots_faced", "sum"), lnh=("lnh_score", "sum"), gk=("is_gk", "max")).reset_index()
    g["lnh"] = g["lnh"].round(1)
    g["pct"] = np.where(g.shots > 0, 100 * g.goals / g.shots.clip(lower=1), 0).round(1)
    g["save_pct"] = np.where(g.faced > 0, 100 * g.saves / g.faced.clip(lower=1), 0).round(1)
    cols = ["name", "team", "games", "goals", "pct", "saves", "faced", "save_pct", "lnh"]
    rec = lambda d: d[cols].to_dict("records")
    return {
        "mvp": rec(g.sort_values("lnh", ascending=False).head(10)),
        "scorers": rec(g[~g.gk].sort_values("goals", ascending=False).head(10)),
        "keepers": rec(g[g.gk & (g.faced >= (60 if season else 150))].sort_values("save_pct", ascending=False).head(10)),
    }


def build_history() -> dict:
    m, box, seasons = load_data()
    pl = m[m.played].copy()
    pl["tot"] = pl.home_ft_goals + pl.away_ft_goals
    per, ranks, teams_by = [], {}, {}
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
    allst = standings(m)
    allst = allst.assign(gf_m=(allst.gf / allst.j).round(2), ga_m=(allst.ga / allst.j).round(2))
    teams_by["all"] = allst[["t", "gf_m", "ga_m", "pts", "j", "rank"]].rename(columns={"t": "team"}).to_dict("records")
    players = {"all": top_players(box, None)}
    players.update({s: top_players(box, s) for s in seasons})
    big = pl.sort_values("tot", ascending=False).iloc[0]
    return {
        "kpis": {"seasons": len(per), "matches": int(len(pl)), "goals": round(pl.tot.mean(), 2),
                 "home_win": round(100 * (pl.home_ft_goals > pl.away_ft_goals).mean(), 1),
                 "home_adv": round((pl.home_ft_goals - pl.away_ft_goals).mean(), 2),
                 "record": {"total": int(big.tot), "score": f"{int(big.home_ft_goals)}-{int(big.away_ft_goals)}",
                            "match": f"{big.home_team} – {big.away_team}", "season": big.season}},
        "per_season": per, "ranks": ranks, "teams": teams_by, "players": players, "seasons": seasons,
    }


def meta() -> dict:
    m, _, seasons = load_data()
    info = {s: {"max_round": int(m[m.season == s]["round"].max()),
                "played_round": int(m[(m.season == s) & m.played]["round"].max()) if m[(m.season == s) & m.played].shape[0] else 0,
                "unplayed": int((~m[m.season == s].played).sum())} for s in seasons}
    return {"seasons": seasons, "info": info, "default": seasons[-1]}


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
    # simulation la plus aberrante = celle qui s'éloigne le plus des points espérés
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

    def send(self, obj, code=200, ctype="application/json"):
        b = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False, default=float).encode()
        self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

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
            if parts[:2] == ["api", "runs"]: return self.send(list_runs())
            if parts[:2] == ["api", "jobs"]: return self.send(JOBS.get(parts[2], {"status": "unknown"}))
            if parts[:2] == ["api", "sim"]:
                if len(parts) > 3 and parts[3] == "upcoming":
                    return self.send((run_dir(parts[2]) / "upcoming_scores.json").read_bytes())
                return self.send(api_sim(parts[2]))
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
