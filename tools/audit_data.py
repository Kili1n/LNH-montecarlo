#!/usr/bin/env python3
"""Audit d'intégrité des CSV (data/data_csv). Travaille sur les fichiers bruts, sans réutiliser le code du serveur.
Usage : python tools/audit_data.py [--json sortie.json]
Affiche, saison par saison, les écarts de cohérence (comptes complets, pas des exemples)."""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "data_csv"
STATS = ["goals", "shots", "goals_7m", "shots_7m", "saves", "shots_faced", "saves_7m", "shots_faced_7m",
         "br", "tc", "pp", "2p", "bp", "dp", "a", "2min", "d"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    return re.sub(r"[^a-z0-9]+", " ", "".join(c for c in s if not unicodedata.combining(c)).lower()).strip()


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    M, B = [], []
    for d in sorted(p for p in DATA.iterdir() if p.is_dir() and re.match(r"\d{4}-\d{4}$", p.name)):
        m = pd.read_csv(d / "match.csv", encoding="utf-8-sig", dtype={"match_id": str})
        m["folder"] = d.name
        M.append(m)
        if (d / "players_boxscore.csv").is_file():
            b = pd.read_csv(d / "players_boxscore.csv", encoding="utf-8-sig", dtype={"match_id": str})
            b["folder"] = d.name
            B.append(b)
    m, b = pd.concat(M, ignore_index=True), pd.concat(B, ignore_index=True)
    res: dict = {}

    # ---- colonnes, types, manquants ---------------------------------------------------------------------------------
    need_m = ["match_id", "season", "round", "date", "home_team", "away_team", "home_ft_goals", "away_ft_goals"]
    need_b = ["match_id", "season", "team", "last_name", "first_name", "seconds_played"] + STATS
    res["colonnes_manquantes"] = {"match": [c for c in need_m if c not in m], "box": [c for c in need_b if c not in b]}
    m["played"] = m["home_ft_goals"].notna() & m["away_ft_goals"].notna()
    res["match_lignes"] = len(m)
    res["box_lignes"] = len(b)
    res["manquants_match_par_colonne"] = {c: int(m[c].isna().sum()) for c in need_m if c in m and m[c].isna().any()}
    res["manquants_box_par_colonne"] = {c: int(b[c].isna().sum()) for c in need_b if c in b and b[c].isna().any()}
    res["season_vs_dossier"] = int((m["season"].str.replace("/", "-") != m["folder"]).sum() + (b["season"].str.replace("/", "-") != b["folder"]).sum())

    # ---- couverture des statistiques par saison (colonne « vide » = non relevée) ----------------------------------------
    cov = b.groupby("season")[STATS].agg(lambda s: float((s.fillna(0) != 0).mean())).round(3)
    res["couverture_stats"] = {s: {c: v for c, v in cov.loc[s].items() if v < 0.005} for s in cov.index}   # colonnes quasi nulles

    # ---- doublons ---------------------------------------------------------------------------------------------------------
    res["doublons_match_season_id"] = int(m.duplicated(["season", "match_id"]).sum())
    res["doublons_match_id_inter_saisons"] = int(m.drop_duplicates(["season", "match_id"]).duplicated("match_id").sum())
    res["doublons_match_journee_equipes"] = int(m.duplicated(["season", "round", "home_team", "away_team"]).sum())
    res["doublons_box"] = int(b.duplicated(["season", "match_id", "team", "last_name", "first_name"]).sum())
    res["doublons_box_inter_saisons_id"] = int((b.groupby("match_id")["season"].nunique() > 1).sum())

    # ---- orphelins ----------------------------------------------------------------------------------------------------------
    km = set(zip(m["season"], m["match_id"]))
    kb = set(zip(b["season"], b["match_id"]))
    res["box_sans_match"] = len(kb - km)
    mp = m[m["played"]]
    sm = {(s, i) for s, i in zip(mp["season"], mp["match_id"])}
    res["matchs_joues_sans_feuille"] = {s: int(sum(1 for x in sm - kb if x[0] == s)) for s in sorted({x[0] for x in sm - kb})}

    # ---- cohérence interne par équipe et match -------------------------------------------------------------------------------
    g = b.groupby(["season", "match_id", "team"]).agg(goals=("goals", "sum"), n=("goals", "size"), secs=("seconds_played", "sum"),
                                                    sf=("shots_faced", "sum"), sv=("saves", "sum")).reset_index()
    h = m[["season", "match_id", "home_team", "home_ft_goals"]].rename(columns={"home_team": "team", "home_ft_goals": "gf"})
    a = m[["season", "match_id", "away_team", "away_ft_goals"]].rename(columns={"away_team": "team", "away_ft_goals": "gf"})
    sc = pd.concat([h, a]).dropna(subset=["gf"])
    j = g.merge(sc, on=["season", "match_id", "team"], how="left")
    res["equipes_box_sans_score_match"] = int(j["gf"].isna().sum())     # équipe de la feuille absente du match (nom de club différent ?)
    j = j.dropna(subset=["gf"])
    j["ecart"] = j["goals"] - j["gf"]
    bad = j[j["ecart"] != 0]
    res["somme_buts_joueurs_diff_score"] = {"equipes_matchs": int(len(bad)), "par_saison": bad.groupby("season").size().to_dict(),
                                             "ecart_moyen_abs": float(bad["ecart"].abs().mean()) if len(bad) else 0.0,
                                             "ecart_max_abs": int(bad["ecart"].abs().max()) if len(bad) else 0}
    res["equipes_sans_aucune_ligne_joueur"] = int(len(sm) - len(kb & sm))

    rules = {
        "buts>tirs": b["goals"] > b["shots"],
        "buts7m>tirs7m": b["goals_7m"] > b["shots_7m"],
        "tirs7m>tirs": b["shots_7m"] > b["shots"],
        "buts7m>buts": b["goals_7m"] > b["goals"],
        "arrets>tirs_subis": b["saves"] > b["shots_faced"],
        "arrets7m>tirs7m_subis": b["saves_7m"] > b["shots_faced_7m"],
        "tirs7m_subis>tirs_subis": b["shots_faced_7m"] > b["shots_faced"],
        "negatif": (b[STATS + ["seconds_played"]] < 0).any(axis=1),
        "minutes>70": b["seconds_played"] > 70 * 60,
        "minutes_nulles_mais_stats": (b["seconds_played"] == 0) & (b[["goals", "shots", "saves", "shots_faced"]].sum(axis=1) > 0),
    }
    res["regles_joueur"] = {k: {"total": int(v.sum()), "par_saison": v.groupby(b["season"]).sum().loc[lambda s: s > 0].astype(int).to_dict()} for k, v in rules.items()}
    # temps de jeu cumulé par équipe plausible (≈ 6 × 3600 s ; 60 min + prolongations) lorsque le chrono existe
    t = g[g["secs"] > 0]
    res["temps_equipe_hors_[18000;26000]s"] = int(((t["secs"] < 18000) | (t["secs"] > 26000)).sum())
    res["lignes_avec_chrono_par_saison"] = b.assign(c=b["seconds_played"] > 0).groupby("season")["c"].mean().round(2).to_dict()

    # ---- un joueur dans deux clubs le même match -------------------------------------------------------------------------------
    b["nom"] = (b["first_name"].fillna("") + " " + b["last_name"].fillna("")).str.strip()
    two = b.groupby(["season", "match_id", "nom"])["team"].nunique()
    res["joueur_deux_clubs_meme_match"] = int((two > 1).sum())
    same = b.groupby(["season", "match_id", "team", "nom"]).size()
    res["joueur_ligne_multiple_meme_match_club"] = int((same > 1).sum())

    # ---- dates ----------------------------------------------------------------------------------------------------------------------
    dt = pd.to_datetime(m["date"], errors="coerce", format="%Y-%m-%d")
    res["dates"] = {"manquantes_total": int(m["date"].isna().sum()), "manquantes_matchs_joues": int((m["played"] & m["date"].isna()).sum()),
                    "non_ISO": int((m["date"].notna() & dt.isna()).sum()), "formats": sorted({re.sub(r"\d", "9", str(x)) for x in m["date"].dropna()})}
    m["_d"] = dt
    inv = 0
    per = {}
    for (s, ), grp in m[m["played"]].groupby(["season"]):
        gg = grp.dropna(subset=["_d"]).sort_values("round").groupby("round")["_d"].agg(["min", "max"])
        # une journée ne devrait pas démarrer avant la fin de la précédente (hors reports) : on compte les chevauchements > 14 jours
        bad_r = int(((gg["min"].shift(-1) < gg["max"] - pd.Timedelta(days=14))).sum())
        if bad_r:
            per[s] = bad_r
        inv += bad_r
    res["journees_dates_non_croissantes(>14j)"] = {"total": inv, "par_saison": per}
    res["matchs_joues_dans_le_futur"] = int(((m["_d"] > pd.Timestamp.today()) & m["played"]).sum())

    # ---- clubs : variantes d'écriture ------------------------------------------------------------------------------------------------
    names_m = set(m["home_team"]) | set(m["away_team"])
    names_b = set(b["team"])
    groups: dict[str, set] = {}
    for n in names_m | names_b:
        groups.setdefault(norm(n), set()).add(n)
    res["clubs_variantes_meme_cle"] = {k: sorted(v) for k, v in groups.items() if len(v) > 1}
    res["clubs_dans_box_pas_dans_match"] = sorted(names_b - names_m)
    res["nb_clubs_distincts_match"] = len(names_m)

    # ---- joueurs : homonymes / casse / accents -------------------------------------------------------------------------------------
    key = b["nom"].map(norm)
    variantes = b.groupby(key)["nom"].nunique()
    res["joueurs_variantes_graphie"] = {k: sorted(b.loc[key == k, "nom"].unique()) for k in variantes[variantes > 1].index[:40]}
    res["joueurs_variantes_graphie_total"] = int((variantes > 1).sum())
    # homonymes certains : même nom, deux lignes dans le même match (deux clubs ou deux fois dans le même club)
    dup_same_match = b.groupby(["season", "match_id", "nom"]).size()
    res["homonymes_meme_match"] = sorted({n for (_, _, n), c in dup_same_match.items() if c > 1})[:40]
    # même nom dans deux clubs la même saison à des dates qui se chevauchent
    ov = b.groupby(["season", "nom"])["team"].nunique()
    res["meme_nom_2_clubs_meme_saison"] = int((ov > 1).sum())

    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res


if __name__ == "__main__":
    main()
