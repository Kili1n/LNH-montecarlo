#!/usr/bin/env python3
"""
LNH Starligue : simulateur multi-saisons Dixon & Coles adapté au handball (offRTG / defRTG).
Aucune notion de possession : offRTG / defRTG sont exprimés en BUTS PAR MATCH.
Exporte automatiquement les résultats dans ./output (Parquet ZSTD + JSON) pour le datastorytelling.
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import logging
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

logger = logging.getLogger("handball_dc_sim")

REQUIRED_MATCH_COLS = [
    "match_id", "season", "round", "date", "time",
    "home_team", "away_team", "home_ft_goals", "away_ft_goals",
]
PTS_WIN, PTS_DRAW, PTS_LOSS = 2, 1, 0
H2H_LABELS = ["points confrontations directes", "diff. buts particulière", "buts marqués particuliers"]
GENERAL_LABELS = ["diff. buts générale", "buts marqués généraux", "ordre alphabétique"]

# Quadrature de Gauss-Hermite (7 points) pour l'intégration exacte du bruit de tempo du match
_GH_X = np.array([
    -2.651961356835233, -1.673551628767471, -0.816287882858965, 0.0,
    0.816287882858965, 1.673551628767471, 2.651961356835233,
], dtype=float)
_GH_W = np.array([
    0.0009717812450995, 0.054515582819127, 0.425607252610128, 0.810264617556807,
    0.425607252610128, 0.054515582819127, 0.0009717812450995,
], dtype=float) / math.sqrt(math.pi)


# =========================================================================== #
# 0. Normalisation canonique des noms de clubs LNH Starligue
# =========================================================================== #
def _team_lookup_key(name: str) -> str:
    """Clé normalisée sans accents, casse ni ponctuation pour unifier les variantes de clubs."""
    s = unicodedata.normalize("NFKD", str(name).strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# Clé normalisée -> Nom canonique retenu dans tout le pipeline
DEFAULT_TEAM_ALIASES: dict[str, str] = {
    # Paris Saint-Germain Handball
    "psg": "PSG",
    "psg handball": "PSG",
    "paris": "PSG",
    "paris sg": "PSG",
    "paris saint germain": "PSG",
    "paris saint germain handball": "PSG",
    "paris hb": "PSG",
    "paris handball": "PSG",
    # HBC Nantes
    "hbc nantes": "HBC Nantes",
    "nantes": "HBC Nantes",
    "handball club de nantes": "HBC Nantes",
    "handball club nantes": "HBC Nantes",
    # Montpellier Handball
    "montpellier hb": "Montpellier HB",
    "montpellier": "Montpellier HB",
    "montpellier handball": "Montpellier HB",
    "mhb": "Montpellier HB",
    "montpellier agglomeration handball": "Montpellier HB",
    "mahb": "Montpellier HB",
    # Fenix Toulouse Handball
    "fenix toulouse": "Fenix Toulouse",
    "toulouse": "Fenix Toulouse",
    "fenix toulouse handball": "Fenix Toulouse",
    "toulouse handball": "Fenix Toulouse",
    "fth": "Fenix Toulouse",
    # Chambéry Savoie Mont-Blanc Handball
    "chambery": "Chambéry",
    "chambery savoie": "Chambéry",
    "chambery savoie handball": "Chambéry",
    "chambery savoie mont blanc handball": "Chambéry",
    "chambery smbh": "Chambéry",
    "csmbh": "Chambéry",
    # USAM Nîmes Gard
    "usam nimes": "USAM Nîmes",
    "nimes": "USAM Nîmes",
    "usam nimes gard": "USAM Nîmes",
    "usam": "USAM Nîmes",
    # Saint-Raphaël Var Handball
    "saint raphael": "Saint-Raphaël",
    "st raphael": "Saint-Raphaël",
    "saint raphael var handball": "Saint-Raphaël",
    "st raphael var handball": "Saint-Raphaël",
    "srvhb": "Saint-Raphaël",
    # Pays d'Aix Université Club Handball
    "pauc": "PAUC",
    "pauc handball": "PAUC",
    "aix": "PAUC",
    "aix en provence": "PAUC",
    "pays d aix": "PAUC",
    "pays d aix uc": "PAUC",
    "pays d aix universite club": "PAUC",
    "pays d aix universite club handball": "PAUC",
    # Limoges Handball
    "limoges": "Limoges",
    "limoges handball": "Limoges",
    "limoges hand 87": "Limoges",
    "lh 87": "Limoges",
    "lh87": "Limoges",
    # Dunkerque Handball Grand Littoral
    "usdk": "USDK Dunkerque",
    "dunkerque": "USDK Dunkerque",
    "usdk dunkerque": "USDK Dunkerque",
    "dunkerque handball": "USDK Dunkerque",
    "dunkerque handball grand littoral": "USDK Dunkerque",
    "dunkerque hgl": "USDK Dunkerque",
    # Cesson Rennes Métropole Handball
    "cesson rennes": "Cesson-Rennes",
    "cesson": "Cesson-Rennes",
    "cesson rennes metropole": "Cesson-Rennes",
    "cesson rennes metropole handball": "Cesson-Rennes",
    "cesson rennes mhb": "Cesson-Rennes",
    "crmhb": "Cesson-Rennes",
    # US Ivry Handball
    "ivry": "US Ivry",
    "us ivry": "US Ivry",
    "us ivry handball": "US Ivry",
    # US Créteil Handball
    "creteil": "US Créteil",
    "us creteil": "US Créteil",
    "us creteil handball": "US Créteil",
    # C'Chartres Métropole Handball
    "chartres": "Chartres",
    "c chartres": "Chartres",
    "c chartres metropole handball": "Chartres",
    "chartres metropole handball": "Chartres",
    "chartres mhb": "Chartres",
    "ccmh": "Chartres",
    # Istres Provence Handball
    "istres": "Istres",
    "istres provence": "Istres",
    "istres provence handball": "Istres",
    "istres ouest provence handball": "Istres",
    "ioph": "Istres",
    # Tremblay Handball
    "tremblay": "Tremblay",
    "tremblay en france": "Tremblay",
    "tremblay handball": "Tremblay",
    "tremblay en france handball": "Tremblay",
    "tfhb": "Tremblay",
    # Saran Loiret Handball
    "saran": "Saran",
    "saran loiret": "Saran",
    "saran loiret handball": "Saran",
    "usm saran": "Saran",
    # Sélestat Alsace Handball
    "selestat": "Sélestat",
    "selestat alsace": "Sélestat",
    "selestat alsace handball": "Sélestat",
    "sahb": "Sélestat",
    # Dijon Métropole Handball
    "dijon": "Dijon",
    "dijon metropole": "Dijon",
    "dijon metropole handball": "Dijon",
    "dijon bourgogne handball": "Dijon",
    "dbhb": "Dijon",
    # Grand Nancy Métropole Handball
    "nancy": "Grand Nancy",
    "grand nancy": "Grand Nancy",
    "grand nancy metropole": "Grand Nancy",
    "grand nancy metropole handball": "Grand Nancy",
    "gnmhb": "Grand Nancy",
    # Autres clubs Proligue / Starligue récents
    "massy": "Massy",
    "massy essonne": "Massy",
    "massy essonne handball": "Massy",
    "mehb": "Massy",
    "pontault combault": "Pontault-Combault",
    "pontault": "Pontault-Combault",
    "ums pontault combault": "Pontault-Combault",
    "ums pontault combault handball": "Pontault-Combault",
    "billere": "Billère",
    "billere handball": "Billère",
    "billere pau pyrenees": "Billère",
    "caen": "Caen",
    "caen handball": "Caen",
    "caen hb": "Caen",
    "frontignan": "Frontignan",
    "frontignan thau": "Frontignan",
    "frontignan thau handball": "Frontignan",
    "fthb": "Frontignan",
    "cherbourg": "Cherbourg",
    "js cherbourg": "Cherbourg",
    "js cherbourg manche handball": "Cherbourg",
    "sarrebourg": "Sarrebourg",
    "sarrebourg moselle sud handball": "Sarrebourg",
    "smshb": "Sarrebourg",
    "valence": "Valence",
    "valence handball": "Valence",
    "besancon": "Grand Besançon",
    "grand besancon": "Grand Besançon",
    "grand besancon doubs handball": "Grand Besançon",
    "gbdh": "Grand Besançon",
}


def load_team_aliases(source: Optional[str | Path] = None) -> dict[str, str]:
    """Construit la table d'alias (défauts LNH + éventuel team_aliases.json / team_aliases.csv)."""
    aliases = dict(DEFAULT_TEAM_ALIASES)
    if source is None:
        return aliases
    p = Path(source)
    base_dir = p if p.is_dir() else p.parent
    if not base_dir.exists():
        return aliases

    json_file = base_dir / "team_aliases.json"
    if json_file.is_file():
        raw = json.loads(json_file.read_text(encoding="utf-8"))
        for k, v in raw.items():
            aliases[_team_lookup_key(k)] = str(v).strip()

    csv_file = base_dir / "team_aliases.csv"
    if csv_file.is_file():
        df_al = pd.read_csv(csv_file)
        if df_al.shape[1] >= 2:
            for k, v in zip(df_al.iloc[:, 0].dropna(), df_al.iloc[:, 1].dropna()):
                aliases[_team_lookup_key(k)] = str(v).strip()
    return aliases


def canonical_team_name(name: Any, alias_map: Optional[dict[str, str]] = None) -> Any:
    """Retourne le nom canonique d'une équipe (préserve NaN)."""
    if pd.isna(name):
        return name
    raw = " ".join(str(name).strip().split())
    if not raw:
        return raw
    amap = alias_map if alias_map is not None else DEFAULT_TEAM_ALIASES
    return amap.get(_team_lookup_key(raw), raw)


def normalize_team_series(s: pd.Series, alias_map: Optional[dict[str, str]] = None) -> pd.Series:
    """Normalise une Series pandas de noms d'équipes via un cache unique."""
    amap = alias_map if alias_map is not None else DEFAULT_TEAM_ALIASES
    uniques = s.dropna().unique()
    mapping = {u: canonical_team_name(u, amap) for u in uniques}
    return s.map(mapping).where(s.notna(), s)


# =========================================================================== #
# 1. Paramètres du modèle
# =========================================================================== #
@dataclass(frozen=True)
class ModelParams:
    """Hyperparamètres du modèle Dixon & Coles Handball (modifiables via --param nom=valeur)."""
    # --- Dixon & Coles : pondération temporelle & inter-saisons ---
    dc_xi: float = 0.55                 # décroissance exponentielle par année
    season_reversion: float = 0.72      # rétention inter-saisons (0.72 = 28 % retour à la moyenne)
    dc_ridge_off: float = 8.0           # régularisation L2 attaque
    dc_ridge_def: float = 8.0           # régularisation L2 défense
    # --- Clubs promus (sans historique récent en Starligue) ---
    promoted_off_delta: float = -0.045
    promoted_def_delta: float = 0.045
    calibrate_promoted: bool = True
    # --- Niveau de base (buts marqués par équipe et par match, si historique absent) ---
    base_goals: float = 30.5
    home_adv_off: float = 0.032
    away_adv_off: float = -0.025
    # --- Dispersion handball ---
    disp_fano: float = 0.58
    tempo_sd_ratio: float = 0.055
    min_team_goals: int = 14
    max_team_goals: int = 55
    # --- Mise à jour dynamique intra-saison ---
    online_k_off: float = 0.022
    online_k_def: float = 0.022
    k_boost: float = 1.2
    k_tau: float = 6.0
    # --- Forme récente ---
    form_coef: float = 0.0045
    form_alpha: float = 0.22
    # --- Avantage terrain propre à la salle ---
    team_ha_k: float = 0.0025
    team_ha_clip: float = 0.06
    # --- Repos ---
    rest_coef: float = 0.0035
    rest_min: float = 2.0
    rest_max: float = 7.0
    default_rest: float = 7.0
    # --- Disponibilité des joueurs clés ---
    absence_off_per_share: float = 0.14
    absence_def_per_share: float = 0.16
    absence_decay: float = 0.85
    absence_lookback: int = 3
    absence_min_games: int = 5
    absence_core_presence: float = 0.5
    gk_weight: float = 1.8
    use_roster_turnover: bool = True
    # --- Incertitude Monte Carlo ---
    rating_noise_sd: float = 0.024
    promoted_noise_mult: float = 1.65
    # --- Classement ---
    reapply_h2h: bool = True


def parse_param_overrides(items: Optional[list[str]]) -> dict:
    """Convertit ['dc_xi=0.6', 'reapply_h2h=false'] en dict typé selon ModelParams."""
    defaults = ModelParams()
    valid = {f.name for f in fields(defaults)}
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--param attend nom=valeur (reçu : {it})")
        key, val = it.split("=", 1)
        if key not in valid:
            raise SystemExit(f"Paramètre inconnu : {key}. Valides : {sorted(valid)}")
        cur = getattr(defaults, key)
        if isinstance(cur, bool):
            out[key] = val.strip().lower() in ("1", "true", "yes", "oui")
        elif isinstance(cur, int):
            out[key] = int(val)
        else:
            out[key] = float(val)
    return out


# =========================================================================== #
# 2. Chargement multi-saisons (racine ou sous-dossiers par année) & calendrier
# =========================================================================== #
def _sort_key(name: str) -> str:
    norm = unicodedata.normalize("NFKD", str(name))
    return "".join(c for c in norm if not unicodedata.combining(c)).lower()


def _season_sort_key(season: str) -> tuple[int, str]:
    m = re.search(r"(20\d{2}|19\d{2})", str(season))
    return (int(m.group(1)) if m else 0, str(season))


def _find_csvs(source: str | Path, default_name: str) -> list[Path]:
    """Trouve un CSV à la racine ou dans tous les sous-dossiers d'années (ex. 2021-2022/match.csv)."""
    p = Path(source)
    if p.is_file():
        return [p]
    base_dir = p if p.is_dir() else p.parent
    target_name = default_name if p.is_dir() else p.name
    if not base_dir.exists():
        return []
    direct = base_dir / target_name
    if direct.is_file():
        return [direct]
    return sorted(base_dir.glob(f"*/{target_name}"))


def load_all_matches(
    csv_path: str | Path,
    boxscore_df: Optional[pd.DataFrame] = None,
    alias_map: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Charge et fusionne match.csv, unifie les noms d'équipes (buts par match uniquement)."""
    amap = alias_map if alias_map is not None else load_team_aliases(csv_path)
    files = _find_csvs(csv_path, "match.csv")
    if not files:
        raise FileNotFoundError(f"Aucun fichier match.csv trouvé dans '{csv_path}' ou ses sous-dossiers.")

    dfs = []
    for f in files:
        part = pd.read_csv(f)
        if "season" not in part.columns:
            part["season"] = f.parent.name
        else:
            part["season"] = part["season"].fillna(f.parent.name)
        dfs.append(part)

    df = pd.concat(dfs, ignore_index=True)
    missing = [c for c in REQUIRED_MATCH_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans match.csv : {missing}")

    # Trace des alias réellement rencontrés pour audit et datastorytelling
    raw_names = set(df["home_team"].dropna().astype(str)) | set(df["away_team"].dropna().astype(str))
    aliases_applied = {
        r: canonical_team_name(r, amap)
        for r in sorted(raw_names)
        if canonical_team_name(r, amap) != r
    }
    if aliases_applied:
        logger.info("Noms d'équipes unifiés (%d variantes) : %s", len(aliases_applied), aliases_applied)

    df["home_team"] = normalize_team_series(df["home_team"], amap)
    df["away_team"] = normalize_team_series(df["away_team"], amap)

    df["match_id"] = df["match_id"].astype(str)
    df["season"] = df["season"].astype(str)
    df = df.drop_duplicates(subset=["season", "match_id"], keep="last").reset_index(drop=True)

    for col in ("home_ft_goals", "away_ft_goals", "round"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["round"]).copy()
    df["round"] = df["round"].astype(int)
    df["played"] = df["home_ft_goals"].notna() & df["away_ft_goals"].notna()
    if (df["home_team"] == df["away_team"]).any():
        bad = df[df["home_team"] == df["away_team"]][["season", "match_id", "home_team"]].head(5)
        raise ValueError(f"Match avec la même équipe à domicile et à l'extérieur :\n{bad}")
    df["time"] = df["time"].fillna("20:00").astype(str)

    played_df = df[df["played"]]
    if not played_df.empty:
        logger.info("Buts : %.2f/match en moyenne sur %d matchs joués.",
                    float((played_df["home_ft_goals"] + played_df["away_ft_goals"]).mean()), len(played_df))
    df["raw_home_off_rtg"] = np.where(df["played"], df["home_ft_goals"], np.nan)
    df["raw_away_off_rtg"] = np.where(df["played"], df["away_ft_goals"], np.nan)

    seasons = sorted(df["season"].dropna().unique(), key=_season_sort_key)
    season_rank = {s: idx for idx, s in enumerate(seasons)}
    df["season_idx"] = df["season"].map(season_rank).astype(int)

    date_dt = pd.to_datetime(df["date"], errors="coerce")
    df["_sort_day"] = date_dt.map(lambda x: float(x.toordinal()) if pd.notna(x) else np.nan)
    df = df.sort_values(["season_idx", "round", "_sort_day", "time", "match_id"], kind="mergesort")
    df = df.drop(columns=["_sort_day"]).reset_index(drop=True)
    return df, seasons, aliases_applied


def check_season_completeness(matches: pd.DataFrame, teams: list[str]) -> None:
    n = len(teams)
    expected = n * (n - 1)
    if len(matches) != expected:
        logger.warning(
            "Saison incomplète dans le CSV : %d matchs présents pour %d attendus (%d équipes, "
            "aller-retour). Utilisez --fill-missing pour générer les matchs manquants.",
            len(matches), expected, n)
    counts = pd.concat([matches["home_team"], matches["away_team"]]).value_counts()
    if counts.nunique() > 1:
        logger.warning("Nombre de matchs inégal selon les équipes (min=%d, max=%d).",
                       counts.min(), counts.max())


def fill_missing_fixtures(matches: pd.DataFrame, teams: list[str], seed: int = 0) -> pd.DataFrame:
    """Ajoute les matchs absents d'un double round-robin (chaque (dom, ext) une fois)."""
    n = len(teams)
    n_rounds, half = 2 * (n - 1), n - 1
    present = set(zip(matches["home_team"], matches["away_team"]))
    missing = [(h, a) for h in teams for a in teams if h != a and (h, a) not in present]
    if not missing:
        return matches
    rng = random.Random(seed)
    rng.shuffle(missing)
    busy: dict[int, set] = {}
    round_of: dict[tuple, int] = {}
    for r, h, a in zip(matches["round"], matches["home_team"], matches["away_team"]):
        busy.setdefault(int(r), set()).update((h, a))
        round_of[(h, a)] = int(r)
    season = matches["season"].iloc[0]
    season_idx = int(matches["season_idx"].iloc[0]) if "season_idx" in matches.columns else 0
    rows = []
    for k, (h, a) in enumerate(missing):
        candidates = list(range(1, n_rounds + 1))
        rng.shuffle(candidates)
        rev = round_of.get((a, h))
        if rev is not None:
            prefer_first_half = rev > half
            candidates.sort(key=lambda r: 0 if ((r <= half) == prefer_first_half) else 1)
        chosen = next((r for r in candidates
                       if h not in busy.get(r, set()) and a not in busy.get(r, set())), None)
        if chosen is None:
            chosen = n_rounds + 1
            while h in busy.get(chosen, set()) or a in busy.get(chosen, set()):
                chosen += 1
        busy.setdefault(chosen, set()).update((h, a))
        round_of[(h, a)] = chosen
        rows.append({
            "match_id": f"FILL{k:03d}", "season": season, "season_idx": season_idx,
            "round": chosen, "date": None, "time": "20:00",
            "home_team": h, "away_team": a,
            "home_ft_goals": np.nan, "away_ft_goals": np.nan, "played": False,
            
        })
    logger.info("%d matchs manquants générés (calendrier complété).", len(rows))
    out = pd.concat([matches, pd.DataFrame(rows)], ignore_index=True)
    out["played"] = out["played"].astype(bool)
    return out


def prepare_fixtures(matches: pd.DataFrame, teams: list[str], cutoff_round: int,
                     p: ModelParams) -> pd.DataFrame:
    """Prépare le calendrier de la saison cible : indices, dates estimées, repos, matchs figés."""
    df = matches.copy()
    idx = {t: i for i, t in enumerate(teams)}
    df["h_idx"] = df["home_team"].map(idx).astype(int)
    df["a_idx"] = df["away_team"].map(idx).astype(int)
    df["time"] = df["time"].fillna("20:00").astype(str)
    df["match_id"] = df["match_id"].astype(str)

    date_dt = pd.to_datetime(df["date"], errors="coerce")
    df["date_estimated"] = date_dt.isna()
    df["day"] = date_dt.map(lambda x: float(x.toordinal()) if pd.notna(x) else np.nan)
    rmin, rmax = int(df["round"].min()), int(df["round"].max())
    rd = df.groupby("round")["day"].median().reindex(range(rmin, rmax + 1))
    rd = rd.interpolate(limit_direction="both")
    if rd.isna().all():
        rd = pd.Series(np.arange(len(rd)) * 7.0 + 1.0, index=rd.index)
    df["day"] = df["day"].fillna(df["round"].map(rd))
    est = df["day"].map(lambda d: dt.date.fromordinal(max(int(round(d)), 1)).isoformat())
    df["date"] = df["date"].where(df["date"].notna(), est)

    df = df.sort_values(["round", "day", "time", "match_id"], kind="mergesort").reset_index(drop=True)

    n = len(df)
    pos = np.arange(n)

    def _side(col: str, side: int) -> pd.DataFrame:
        return pd.DataFrame({"pos": pos, "team": df[col].to_numpy(), "side": side,
                             "day": df["day"].to_numpy()})

    long = pd.concat([_side("h_idx", 0), _side("a_idx", 1)], ignore_index=True)
    long = long.sort_values(["team", "pos"], kind="mergesort")
    grp = long.groupby("team")
    long["rest"] = grp["day"].diff()
    long["n_prior"] = grp.cumcount()
    long["rest"] = long["rest"].fillna(p.default_rest).clip(lower=0.0)
    hh = long[long["side"] == 0].sort_values("pos")
    aa = long[long["side"] == 1].sort_values("pos")
    df["rest_h"], df["rest_a"] = hh["rest"].to_numpy(), aa["rest"].to_numpy()
    df["n_prior_h"], df["n_prior_a"] = hh["n_prior"].to_numpy(), aa["n_prior"].to_numpy()
    df["rest_diff"] = (np.clip(df["rest_h"], p.rest_min, p.rest_max)
                       - np.clip(df["rest_a"], p.rest_min, p.rest_max))

    df["frozen"] = df["played"] & (df["round"] <= cutoff_round)
    df["avail_off_h"] = 0.0
    df["avail_off_a"] = 0.0
    df["avail_def_h"] = 0.0
    df["avail_def_a"] = 0.0
    return df


# =========================================================================== #
# 3. Moteur Dixon & Coles Multi-Saisons (offRTG, defRTG, Promus)
# =========================================================================== #
@dataclass
class DixonColesFit:
    """Résultat de l'estimation Dixon & Coles pondérée sur l'historique."""
    teams: list[str]
    alpha: np.ndarray
    beta: np.ndarray
    mu_eff: float
    home_adv_off: float
    away_adv_off: float
    is_promoted: np.ndarray
    hist_matches: np.ndarray
    raw_off_rtg: np.ndarray
    raw_def_rtg: np.ndarray
    promoted_off_used: float
    promoted_def_used: float

    @property
    def off_rtg(self) -> np.ndarray:
        return np.exp(self.mu_eff + self.alpha)  # buts marqués / match

    @property
    def def_rtg(self) -> np.ndarray:
        return np.exp(self.mu_eff + self.beta)  # buts encaissés / match

    @property
    def net_rtg(self) -> np.ndarray:
        return self.off_rtg - self.def_rtg

    def summary_table(self) -> pd.DataFrame:
        df = pd.DataFrame({
            "Team": self.teams,
            "offRTG": self.off_rtg,
            "defRTG": self.def_rtg,
            "netRTG": self.net_rtg,
            "Raw_offRTG": self.raw_off_rtg,
            "Raw_defRTG": self.raw_def_rtg,
            "Promoted": self.is_promoted,
            "Hist_Matches": np.round(self.hist_matches, 1),
        })
        return df.sort_values("netRTG", ascending=False).reset_index(drop=True)


def calibrate_promoted_priors(all_matches: pd.DataFrame, target_season_idx: int,
                              p: ModelParams) -> tuple[float, float]:
    """Estime empiriquement le décalage (alpha_prom, beta_prom) des clubs promus en Starligue."""
    if not p.calibrate_promoted or target_season_idx <= 0:
        return p.promoted_off_delta, p.promoted_def_delta

    hist = all_matches[all_matches["played"] & (all_matches["season_idx"] < target_season_idx)]
    if hist.empty:
        return p.promoted_off_delta, p.promoted_def_delta

    seasons_present = sorted(hist["season_idx"].unique())
    if len(seasons_present) < 2:
        return p.promoted_off_delta, p.promoted_def_delta

    off_deltas, def_deltas = [], []
    teams_by_season: dict[int, set[str]] = {}
    for s_idx, grp in hist.groupby("season_idx"):
        teams_by_season[int(s_idx)] = set(grp["home_team"]) | set(grp["away_team"])

    for idx_pos in range(1, len(seasons_present)):
        s_curr = seasons_present[idx_pos]
        s_prev = seasons_present[idx_pos - 1]
        promoted = teams_by_season[s_curr] - teams_by_season[s_prev]
        if not promoted:
            continue
        sg = hist[hist["season_idx"] == s_curr]
        league_eff = float((sg["home_ft_goals"].sum() + sg["away_ft_goals"].sum()) / (2.0 * len(sg)))
        if league_eff <= 0:
            continue
        for t in promoted:
            hm = sg[sg["home_team"] == t]
            am = sg[sg["away_team"] == t]
            gf = float(hm["home_ft_goals"].sum() + am["away_ft_goals"].sum())
            ga = float(hm["away_ft_goals"].sum() + am["home_ft_goals"].sum())
            n_g = float(len(hm) + len(am))
            if n_g >= 8.0 and gf > 0 and ga > 0:
                off_deltas.append(math.log((gf / n_g) / league_eff))
                def_deltas.append(math.log((ga / n_g) / league_eff))

    if not off_deltas:
        return p.promoted_off_delta, p.promoted_def_delta

    n_prom = len(off_deltas)
    w_prior = 6.0
    prom_off = (n_prom * float(np.mean(off_deltas)) + w_prior * p.promoted_off_delta) / (n_prom + w_prior)
    prom_def = (n_prom * float(np.mean(def_deltas)) + w_prior * p.promoted_def_delta) / (n_prom + w_prior)
    return float(np.clip(prom_off, -0.15, 0.05)), float(np.clip(prom_def, -0.05, 0.18))


def _solve_weighted_poisson_dc(
    h_idx: np.ndarray,
    a_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    prior_alpha: np.ndarray,
    prior_beta: np.ndarray,
    ridge_off: float,
    ridge_def: float,
    n_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Résout le MLE pondéré Dixon & Coles pour l'attaque et la défense via Newton-Raphson."""
    alpha = prior_alpha.copy()
    beta = prior_beta.copy()
    tot_goals = float(np.sum(weights * (hg + ag)))
    tot_matches2 = float(np.sum(weights) * 2.0)
    mu_eff = math.log(max(tot_goals / max(tot_matches2, 1e-6), 0.2))
    home_off = 0.03
    away_off = -0.025

    w_hg = weights * hg
    w_ag = weights * ag

    for _ in range(n_iter):
        pred_h = np.exp(mu_eff + alpha[h_idx] + beta[a_idx] + home_off)
        pred_a = np.exp(mu_eff + alpha[a_idx] + beta[h_idx] + away_off)
        d_home = (np.sum(w_hg) - np.sum(weights * pred_h)) / max(np.sum(weights * pred_h), 1e-6)
        d_away = (np.sum(w_ag) - np.sum(weights * pred_a)) / max(np.sum(weights * pred_a), 1e-6)
        home_off += float(np.clip(d_home, -0.2, 0.2))
        away_off += float(np.clip(d_away, -0.2, 0.2))
        shift = 0.5 * (home_off + away_off)
        mu_eff += shift
        home_off -= shift
        away_off -= shift

        pred_h = np.exp(mu_eff + alpha[h_idx] + beta[a_idx] + home_off)
        pred_a = np.exp(mu_eff + alpha[a_idx] + beta[h_idx] + away_off)
        grad_a = (np.bincount(h_idx, w_hg - weights * pred_h, n_teams)
                  + np.bincount(a_idx, w_ag - weights * pred_a, n_teams)
                  - ridge_off * (alpha - prior_alpha))
        hess_a = (np.bincount(h_idx, weights * pred_h, n_teams)
                  + np.bincount(a_idx, weights * pred_a, n_teams)
                  + ridge_off)
        alpha += np.clip(grad_a / np.maximum(hess_a, 1e-6), -0.25, 0.25)

        pred_h = np.exp(mu_eff + alpha[h_idx] + beta[a_idx] + home_off)
        pred_a = np.exp(mu_eff + alpha[a_idx] + beta[h_idx] + away_off)
        grad_b = (np.bincount(a_idx, w_hg - weights * pred_h, n_teams)
                  + np.bincount(h_idx, w_ag - weights * pred_a, n_teams)
                  - ridge_def * (beta - prior_beta))
        hess_b = (np.bincount(a_idx, weights * pred_h, n_teams)
                  + np.bincount(h_idx, weights * pred_a, n_teams)
                  + ridge_def)
        beta += np.clip(grad_b / np.maximum(hess_b, 1e-6), -0.25, 0.25)

        alpha -= (np.mean(alpha) - np.mean(prior_alpha))
        beta -= (np.mean(beta) - np.mean(prior_beta))

    return alpha, beta, mu_eff, home_off, away_off


def fit_dixon_coles_ratings(
    all_matches: pd.DataFrame,
    target_season: str,
    target_teams: list[str],
    cutoff_round: int,
    p: ModelParams,
    roster_retention_adj: Optional[dict[str, float]] = None,
) -> DixonColesFit:
    """Calcule les ratings Dixon & Coles (offRTG, defRTG) sur l'historique pondéré."""
    season_rows = all_matches[all_matches["season"] == target_season]
    if season_rows.empty:
        raise ValueError(f"Saison cible '{target_season}' introuvable dans les données.")
    target_s_idx = int(season_rows["season_idx"].iloc[0])

    hist = all_matches[
        all_matches["played"]
        & (
            (all_matches["season_idx"] < target_s_idx)
            | ((all_matches["season_idx"] == target_s_idx) & (all_matches["round"] <= cutoff_round))
        )
    ].copy()

    prom_off, prom_def = calibrate_promoted_priors(all_matches, target_s_idx, p)

    all_team_names = sorted(
        set(target_teams) | set(hist["home_team"]) | set(hist["away_team"]),
        key=_sort_key,
    )
    t_map = {t: i for i, t in enumerate(all_team_names)}
    n_all = len(all_team_names)

    past_seasons_df = hist[hist["season_idx"] < target_s_idx]
    established_teams = set(past_seasons_df["home_team"]) | set(past_seasons_df["away_team"])
    is_prom_target = np.array([t not in established_teams for t in target_teams], dtype=bool)

    prior_alpha = np.zeros(n_all, dtype=float)
    prior_beta = np.zeros(n_all, dtype=float)
    for t in target_teams:
        if t not in established_teams:
            prior_alpha[t_map[t]] = prom_off
            prior_beta[t_map[t]] = prom_def

    if hist.empty:
        target_indices = [t_map[t] for t in target_teams]
        return DixonColesFit(
            teams=list(target_teams),
            alpha=prior_alpha[target_indices],
            beta=prior_beta[target_indices],
            mu_eff=math.log(p.base_goals),
            home_adv_off=p.home_adv_off,
            away_adv_off=p.away_adv_off,
            is_promoted=is_prom_target,
            hist_matches=np.zeros(len(target_teams)),
            raw_off_rtg=np.full(len(target_teams), p.base_goals),
            raw_def_rtg=np.full(len(target_teams), p.base_goals),
            promoted_off_used=prom_off,
            promoted_def_used=prom_def,
        )

    hist_days = pd.to_datetime(hist["date"], errors="coerce").map(
        lambda x: float(x.toordinal()) if pd.notna(x) else np.nan
    )
    target_days = pd.to_datetime(season_rows["date"], errors="coerce").map(
        lambda x: float(x.toordinal()) if pd.notna(x) else np.nan
    )
    if cutoff_round > 0 and (season_rows["round"] <= cutoff_round).any():
        ref_day = float(target_days[season_rows["round"] <= cutoff_round].max())
    elif target_days.notna().any():
        ref_day = float(target_days.min())
    elif hist_days.notna().any():
        ref_day = float(hist_days.max())
    else:
        ref_day = 0.0

    hist_days = hist_days.fillna(ref_day).to_numpy(dtype=float)
    dt_years = np.maximum(0.0, (ref_day - hist_days) / 365.25)
    delta_seasons = np.maximum(0, target_s_idx - hist["season_idx"].to_numpy(dtype=int))

    base_rev = np.full(len(hist), p.season_reversion, dtype=float)
    if roster_retention_adj:
        h_adj = np.array([roster_retention_adj.get(t, p.season_reversion) for t in hist["home_team"]])
        a_adj = np.array([roster_retention_adj.get(t, p.season_reversion) for t in hist["away_team"]])
        pair_rev = 0.5 * (h_adj + a_adj)
        base_rev = np.where(delta_seasons > 0, pair_rev, 1.0)

    weights = np.exp(-p.dc_xi * dt_years) * (base_rev ** delta_seasons)
    weights = np.clip(weights, 1e-4, 1.0)

    h_idx = hist["home_team"].map(t_map).to_numpy(dtype=int)
    a_idx = hist["away_team"].map(t_map).to_numpy(dtype=int)
    hg = hist["home_ft_goals"].to_numpy(dtype=float)
    ag = hist["away_ft_goals"].to_numpy(dtype=float)

    alpha_all, beta_all, mu_eff, home_off, away_off = _solve_weighted_poisson_dc(
        h_idx, a_idx, hg, ag, weights, n_all,
        prior_alpha, prior_beta, p.dc_ridge_off, p.dc_ridge_def,
    )
    target_indices = [t_map[t] for t in target_teams]
    w_matches = np.bincount(h_idx, weights, n_all) + np.bincount(a_idx, weights, n_all)
    w_gf = np.bincount(h_idx, weights * hg, n_all) + np.bincount(a_idx, weights * ag, n_all)
    w_ga = np.bincount(h_idx, weights * ag, n_all) + np.bincount(a_idx, weights * hg, n_all)

    ok = w_matches[target_indices] > 1.0
    raw_off = np.where(ok, w_gf[target_indices] / np.maximum(w_matches[target_indices], 1.0),
                       np.exp(mu_eff + prior_alpha[target_indices]))
    raw_def = np.where(ok, w_ga[target_indices] / np.maximum(w_matches[target_indices], 1.0),
                       np.exp(mu_eff + prior_beta[target_indices]))

    return DixonColesFit(
        teams=list(target_teams),
        alpha=alpha_all[target_indices],
        beta=beta_all[target_indices],
        mu_eff=mu_eff,
        home_adv_off=home_off,
        away_adv_off=away_off,
        is_promoted=is_prom_target,
        hist_matches=w_matches[target_indices],
        raw_off_rtg=raw_off,
        raw_def_rtg=raw_def,
        promoted_off_used=prom_off,
        promoted_def_used=prom_def,
    )


# =========================================================================== #
# 4. Données joueurs : disponibilité (attaque vs défense/GB) & mercato
# =========================================================================== #
def load_player_tables(
    boxscore_csv: str | Path,
    players_csv: Optional[str | Path],
    alias_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Charge et concatène players_boxscore.csv et players.csv en unifiant les noms de clubs."""
    amap = alias_map if alias_map is not None else load_team_aliases(boxscore_csv)
    box_files = _find_csvs(boxscore_csv, "players_boxscore.csv")
    if not box_files:
        raise FileNotFoundError(f"Aucun fichier players_boxscore.csv trouvé dans '{boxscore_csv}'.")

    box = pd.concat([pd.read_csv(f).assign(season=lambda d, f=f: d['season'] if 'season' in d.columns else f.parent.name)
                     for f in box_files], ignore_index=True)
    if "team" in box.columns:
        box["team"] = normalize_team_series(box["team"], amap)

    num_cols = [
        "seconds_played", "goals", "shots", "goals_7m", "shots_7m",
        "saves", "shots_faced", "saves_7m", "shots_faced_7m",
        "br", "tc", "pp", "2p", "bp", "dp", "a", "2min", "d", "lnh_score",
    ]
    for c in num_cols:
        if c in box.columns:
            box[c] = pd.to_numeric(box[c], errors="coerce").fillna(0.0)
        else:
            box[c] = 0.0
    box["match_id"] = box["match_id"].astype(str)
    box = box.drop_duplicates(subset=["season", "match_id", "team", "last_name", "first_name"], keep="last")

    key = ["team", "last_name", "first_name"]
    player_files = _find_csvs(players_csv, "players.csv") if players_csv is not None else []
    if player_files:
        pl = pd.concat([pd.read_csv(f) for f in player_files], ignore_index=True)
        if "team" in pl.columns:
            pl["team"] = normalize_team_series(pl["team"], amap)
        if "current_club" in pl.columns:
            pl["current_club"] = normalize_team_series(pl["current_club"], amap)
        keep = [c for c in ("position", "current_club") if c in pl.columns]
        if keep:
            box = box.merge(pl[key + keep].drop_duplicates(key, keep="last"), on=key, how="left")
        else:
            box["position"] = np.nan
    else:
        box["position"] = np.nan

    box["is_gk"] = (
        box["position"].fillna("").str.contains("Gardien", case=False)
        | (box["shots_faced"] > 0)
    )
    return box


def compute_roster_turnover(
    box: Optional[pd.DataFrame],
    prev_season_matches: pd.DataFrame,
    teams: list[str],
    p: ModelParams,
) -> dict[str, float]:
    """Estime la rétention d'effectif entre N-1 et N via `current_club` dans players.csv."""
    if not p.use_roster_turnover or box is None or "current_club" not in box.columns or prev_season_matches.empty:
        return {}
    prev_ids = set(prev_season_matches["match_id"].astype(str))
    pb = box[box["match_id"].isin(prev_ids)].copy()
    if pb.empty or pb["current_club"].notna().sum() == 0:
        return {}

    out: dict[str, float] = {}
    for team in teams:
        tb = pb[pb["team"] == team]
        if tb.empty:
            continue
        g = tb.groupby(["last_name", "first_name"]).agg(
            score=("lnh_score", "sum"),
            cur=("current_club", "last"),
        )
        g["score"] = g["score"].clip(lower=0.0)
        tot = float(g["score"].sum())
        if tot <= 0:
            continue
        retained = float(g.loc[g["cur"].fillna("").apply(_sort_key) == _sort_key(team), "score"].sum())
        share_retained = float(np.clip(retained / tot, 0.25, 1.0))
        adj_rev = p.season_reversion * (0.75 + 0.35 * share_retained)
        out[team] = float(np.clip(adj_rev, 0.45, 0.90))
    return out


def compute_availability(
    box: Optional[pd.DataFrame],
    known: pd.DataFrame,
    teams: list[str],
    p: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les malus d'absence (attaque vs défense/gardiens) au moment du cutoff."""
    n = len(teams)
    pen_off = np.zeros(n, dtype=float)
    pen_def = np.zeros(n, dtype=float)
    if box is None or known.empty or (p.absence_off_per_share == 0 and p.absence_def_per_share == 0):
        return pen_off, pen_def

    kb = box[box["match_id"].isin(set(known["match_id"]))]
    for t_idx, team in enumerate(teams):
        tm = known[(known["home_team"] == team) | (known["away_team"] == team)]
        ids = tm["match_id"].tolist()
        if len(ids) < max(p.absence_min_games, p.absence_lookback + 1):
            continue
        recent, past = ids[-p.absence_lookback:], ids[:-p.absence_lookback]
        tb = kb[kb["team"] == team]
        past_b, rec_b = tb[tb["match_id"].isin(past)], tb[tb["match_id"].isin(recent)]
        if past_b.empty or rec_b["match_id"].nunique() < len(recent):
            continue

        raw_off_pts = (
            past_b["goals"]
            + 0.7 * past_b["a"]
            + 0.5 * past_b["dp"]
            - 0.6 * past_b["bp"]
            + 0.35 * past_b["lnh_score"].clip(lower=0.0)
        ).clip(lower=0.0)

        raw_def_pts = np.where(
            past_b["is_gk"],
            (1.2 * past_b["saves"] + 0.5 * past_b["lnh_score"].clip(lower=0.0)).clip(lower=0.0) * p.gk_weight,
            (1.2 * past_b["br"] + 1.0 * past_b["tc"] + 0.35 * past_b["lnh_score"].clip(lower=0.0)).clip(lower=0.0),
        )

        work = past_b.assign(
            played_flag=(past_b["seconds_played"] > 0).astype(int),
            _off=raw_off_pts,
            _def=raw_def_pts,
        )
        g = (
            work.groupby(["last_name", "first_name"])
            .agg(games=("played_flag", "sum"), off_sum=("_off", "sum"), def_sum=("_def", "sum"))
            .reset_index()
        )
        g["presence"] = g["games"] / len(past)
        g["off_val"] = (g["off_sum"] / g["games"].clip(lower=1)) * g["presence"]
        g["def_val"] = (g["def_sum"] / g["games"].clip(lower=1)) * g["presence"]

        tot_off = float(g["off_val"].sum())
        tot_def = float(g["def_val"].sum())
        if tot_off <= 0 and tot_def <= 0:
            continue

        rec_sec = (
            rec_b.groupby(["last_name", "first_name"])["seconds_played"]
            .sum()
            .reset_index()
            .rename(columns={"seconds_played": "rec_sec"})
        )
        g = g.merge(rec_sec, on=["last_name", "first_name"], how="left")
        g["rec_sec"] = g["rec_sec"].fillna(0.0)
        absent = g[(g["presence"] >= p.absence_core_presence) & (g["rec_sec"] <= 0)]

        if tot_off > 0:
            pen_off[t_idx] = -p.absence_off_per_share * float(absent["off_val"].sum()) / tot_off
        if tot_def > 0:
            pen_def[t_idx] = p.absence_def_per_share * float(absent["def_val"].sum()) / tot_def

    return pen_off, pen_def


def apply_availability(fx: pd.DataFrame, pen_off: np.ndarray, pen_def: np.ndarray,
                       cutoff_round: int, p: ModelParams) -> None:
    rounds = fx["round"].to_numpy()
    after = rounds > cutoff_round
    decay = p.absence_decay ** np.maximum(rounds - cutoff_round - 1, 0)
    h = fx["h_idx"].to_numpy()
    a = fx["a_idx"].to_numpy()
    fx["avail_off_h"] = np.where(after, pen_off[h] * decay, 0.0)
    fx["avail_off_a"] = np.where(after, pen_off[a] * decay, 0.0)
    fx["avail_def_h"] = np.where(after, pen_def[h] * decay, 0.0)
    fx["avail_def_a"] = np.where(after, pen_def[a] * decay, 0.0)


# =========================================================================== #
# 5. Briques mathématiques Handball Dixon & Coles
# =========================================================================== #
def _lgamma(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    cof = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5,
    ]
    y = x.copy()
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    ser = np.full_like(x, 1.000000000190015)
    for c in cof:
        y += 1.0
        ser += c / y
    return -tmp + np.log(2.5066282746310005 * ser / x)


def _goal_pmf_matrix(lam: np.ndarray, disp_fano: float, max_g: int = 65) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    k = np.arange(max_g + 1, dtype=float)[None, :]
    log_fact_k = np.zeros(max_g + 1, dtype=float)
    log_fact_k[1:] = np.cumsum(np.log(np.arange(1, max_g + 1, dtype=float)))

    if disp_fano < 0.98:
        p_bin = float(np.clip(1.0 - disp_fano, 0.02, 0.95))
        n_trials = np.maximum(np.rint(lam / p_bin), 1.0)[:, None]
        valid = k <= n_trials
        n_safe = np.maximum(n_trials, k)
        log_pmf = (
            _lgamma(n_safe + 1.0) - log_fact_k[None, :] - _lgamma(n_safe - k + 1.0)
            + k * math.log(p_bin)
            + (n_safe - k) * math.log(1.0 - p_bin)
        )
        pmf = np.where(valid, np.exp(np.clip(log_pmf, -60.0, 0.0)), 0.0)
    elif disp_fano <= 1.02:
        lam_col = np.maximum(lam, 1e-6)[:, None]
        log_pmf = k * np.log(lam_col) - lam_col - log_fact_k[None, :]
        pmf = np.exp(np.clip(log_pmf, -60.0, 0.0))
    else:
        r = np.maximum(lam / (disp_fano - 1.0), 0.5)[:, None]
        prob = r / (r + np.maximum(lam, 1e-6)[:, None])
        log_pmf = (
            _lgamma(k + r) - log_fact_k[None, :] - _lgamma(r)
            + r * np.log(prob)
            + k * np.log(1.0 - prob)
        )
        pmf = np.exp(np.clip(log_pmf, -60.0, 0.0))

    row_sums = pmf.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, pmf / row_sums, 1.0 / (max_g + 1))


def outcome_probs_dc(lam_h: np.ndarray, lam_a: np.ndarray, p: ModelParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lh = np.asarray(lam_h, dtype=float)
    la = np.asarray(lam_a, dtype=float)
    n = len(lh)
    p_home = np.zeros(n, dtype=float)
    p_draw = np.zeros(n, dtype=float)
    p_away = np.zeros(n, dtype=float)

    sd_rel = max(p.tempo_sd_ratio, 0.0)
    nodes = _GH_X if sd_rel > 1e-4 else np.array([0.0])
    weights = _GH_W if sd_rel > 1e-4 else np.array([1.0])

    for z_node, w_node in zip(nodes, weights):
        scale = max(1.0 + math.sqrt(2.0) * sd_rel * z_node, 0.5)
        pmf_h = _goal_pmf_matrix(lh * scale, p.disp_fano)
        pmf_a = _goal_pmf_matrix(la * scale, p.disp_fano)
        cdf_a_strict = np.cumsum(pmf_a, axis=1) - pmf_a
        p_d_node = np.sum(pmf_h * pmf_a, axis=1)
        p_h_node = np.sum(pmf_h * cdf_a_strict, axis=1)
        p_a_node = np.clip(1.0 - p_h_node - p_d_node, 0.0, 1.0)
        p_home += w_node * p_h_node
        p_draw += w_node * p_d_node
        p_away += w_node * p_a_node

    tot = np.maximum(p_home + p_draw + p_away, 1e-12)
    return p_home / tot, p_draw / tot, p_away / tot


def sample_scores_dc(
    rng: np.random.Generator,
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    p: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    if p.tempo_sd_ratio > 1e-4:
        tempo_shock = np.clip(rng.normal(1.0, p.tempo_sd_ratio, size=lam_h.shape), 0.65, 1.45)
    else:
        tempo_shock = 1.0

    lh = lam_h * tempo_shock
    la = lam_a * tempo_shock

    if p.disp_fano < 0.98:
        p_bin = float(np.clip(1.0 - p.disp_fano, 0.02, 0.95))
        nh = np.maximum(np.rint(lh / p_bin).astype(np.int64), 1)
        na = np.maximum(np.rint(la / p_bin).astype(np.int64), 1)
        hg = rng.binomial(nh, p_bin)
        ag = rng.binomial(na, p_bin)
    elif p.disp_fano <= 1.02:
        hg = rng.poisson(np.maximum(lh, 0.1))
        ag = rng.poisson(np.maximum(la, 0.1))
    else:
        r_h = np.maximum(lh / (p.disp_fano - 1.0), 0.5)
        r_a = np.maximum(la / (p.disp_fano - 1.0), 0.5)
        gh_rate = rng.gamma(r_h, lh / r_h)
        ga_rate = rng.gamma(r_a, la / r_a)
        hg = rng.poisson(gh_rate)
        ag = rng.poisson(ga_rate)

    hg = np.clip(hg, p.min_team_goals, p.max_team_goals).astype(np.int64)
    ag = np.clip(ag, p.min_team_goals, p.max_team_goals).astype(np.int64)
    return hg, ag


# =========================================================================== #
# 6. État des équipes, prédiction et mise à jour dynamique
# =========================================================================== #
@dataclass
class RatingState:
    alpha: np.ndarray
    beta: np.ndarray
    form_off: np.ndarray
    form_def: np.ndarray
    ha: np.ndarray

    @classmethod
    def initial(cls, n_sims: int, dc: DixonColesFit) -> "RatingState":
        n_teams = len(dc.teams)
        zeros = np.zeros((n_sims, n_teams), dtype=float)
        return cls(
            alpha=np.tile(dc.alpha, (n_sims, 1)),
            beta=np.tile(dc.beta, (n_sims, 1)),
            form_off=zeros.copy(),
            form_def=zeros.copy(),
            ha=zeros.copy(),
        )

    def copy(self) -> "RatingState":
        return RatingState(
            self.alpha.copy(), self.beta.copy(),
            self.form_off.copy(), self.form_def.copy(), self.ha.copy(),
        )


def predict_dc(
    state: RatingState,
    h: int,
    a: int,
    avail_off_h: float,
    avail_off_a: float,
    avail_def_h: float,
    avail_def_a: float,
    rest_diff: float,
    dc: DixonColesFit,
    p: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:

    log_eff_h = (
        dc.mu_eff
        + state.alpha[:, h]
        + state.beta[:, a]
        + dc.home_adv_off
        + state.ha[:, h]
        + avail_off_h
        + avail_def_a
        + p.rest_coef * rest_diff
        + p.form_coef * (state.form_off[:, h] + state.form_def[:, a])
    )
    log_eff_a = (
        dc.mu_eff
        + state.alpha[:, a]
        + state.beta[:, h]
        + dc.away_adv_off
        - state.ha[:, h]
        + avail_off_a
        + avail_def_h
        - p.rest_coef * rest_diff
        + p.form_coef * (state.form_off[:, a] + state.form_def[:, h])
    )

    lam_h = np.clip(np.exp(log_eff_h), 15.0, 50.0)
    lam_a = np.clip(np.exp(log_eff_a), 15.0, 50.0)
    return lam_h, lam_a


def update_state_dc(
    state: RatingState,
    h: int,
    a: int,
    hg: np.ndarray,
    ag: np.ndarray,
    lam_h: np.ndarray,
    lam_a: np.ndarray,
    n_prior_h: int,
    n_prior_a: int,
    is_prom_h: bool,
    is_prom_a: bool,
    p: ModelParams,
) -> None:
    hg_f = hg.astype(float)
    ag_f = ag.astype(float)

    eff_prior_h = n_prior_h if is_prom_h else (n_prior_h + 10)
    eff_prior_a = n_prior_a if is_prom_a else (n_prior_a + 10)
    boost_h = 1.0 + p.k_boost * math.exp(-eff_prior_h / p.k_tau)
    boost_a = 1.0 + p.k_boost * math.exp(-eff_prior_a / p.k_tau)

    err_h = np.clip((hg_f - lam_h) / np.maximum(lam_h, 15.0), -0.45, 0.45)
    err_a = np.clip((ag_f - lam_a) / np.maximum(lam_a, 15.0), -0.45, 0.45)

    state.alpha[:, h] += p.online_k_off * boost_h * err_h
    state.beta[:, a] += p.online_k_def * boost_a * err_h

    state.alpha[:, a] += p.online_k_off * boost_a * err_a
    state.beta[:, h] += p.online_k_def * boost_h * err_a

    res_h = hg_f - lam_h
    res_a = ag_f - lam_a
    state.form_off[:, h] = (1.0 - p.form_alpha) * state.form_off[:, h] + p.form_alpha * res_h
    state.form_def[:, a] = (1.0 - p.form_alpha) * state.form_def[:, a] + p.form_alpha * res_h
    state.form_off[:, a] = (1.0 - p.form_alpha) * state.form_off[:, a] + p.form_alpha * res_a
    state.form_def[:, h] = (1.0 - p.form_alpha) * state.form_def[:, h] + p.form_alpha * res_a

    margin_err = res_h - res_a
    state.ha[:, h] = np.clip(state.ha[:, h] + p.team_ha_k * margin_err,
                             -p.team_ha_clip, p.team_ha_clip)


# =========================================================================== #
# 7. Historique & calibration des dispersions
# =========================================================================== #
def run_history(
    fx: pd.DataFrame,
    dc: DixonColesFit,
    p: ModelParams,
    cutoff_round: int,
) -> tuple[RatingState, RatingState, pd.DataFrame]:
    state = RatingState.initial(1, dc)
    snapshot: Optional[RatingState] = None

    H, A = fx["h_idx"].to_numpy(), fx["a_idx"].to_numpy()
    HG, AG = fx["home_ft_goals"].to_numpy(), fx["away_ft_goals"].to_numpy()
    played, R = fx["played"].to_numpy(), fx["round"].to_numpy()
    RD = fx["rest_diff"].to_numpy()
    AOH, AOA = fx["avail_off_h"].to_numpy(), fx["avail_off_a"].to_numpy()
    ADH, ADA = fx["avail_def_h"].to_numpy(), fx["avail_def_a"].to_numpy()
    NH, NA = fx["n_prior_h"].to_numpy(), fx["n_prior_a"].to_numpy()
    ids, hn, an = fx["match_id"].to_numpy(), fx["home_team"].to_numpy(), fx["away_team"].to_numpy()

    rows = []
    for i in range(len(fx)):
        if not played[i]:
            continue
        if snapshot is None and R[i] > cutoff_round:
            snapshot = state.copy()
        h, a = int(H[i]), int(A[i])
        lh, la = predict_dc(state, h, a, AOH[i], AOA[i], ADH[i], ADA[i], RD[i], dc, p)
        hg, ag = np.array([int(HG[i])]), np.array([int(AG[i])])
        p_h, p_d, p_a = outcome_probs_dc(lh, la, p)
        rows.append((
            ids[i], int(R[i]), hn[i], an[i],
            float(lh[0] - la[0]), float(lh[0] + la[0]), float(lh[0]), float(la[0]),
            float(p_h[0]), float(p_d[0]), float(p_a[0]), int(hg[0]), int(ag[0]),
        ))
        update_state_dc(
            state, h, a, hg, ag, lh, la,
            int(NH[i]), int(NA[i]), bool(dc.is_promoted[h]), bool(dc.is_promoted[a]), p,
        )

    if snapshot is None:
        snapshot = state.copy()
    preds = pd.DataFrame(rows, columns=[
        "match_id", "round", "home_team", "away_team",
        "exp_margin", "exp_total", "exp_hg", "exp_ag",
        "p_home", "p_draw", "p_away", "hg", "ag",
    ])
    return snapshot, state, preds


def fit_dispersion_params(
    fx: pd.DataFrame,
    dc: DixonColesFit,
    p0: ModelParams,
    cutoff_round: int,
    prior_weight: float = 35.0,
) -> ModelParams:
    known = fx[fx["played"] & (fx["round"] <= cutoff_round)]
    n = len(known)
    if n < 8:
        return p0

    _, _, preds = run_history(known, dc, p0, cutoff_round=10 ** 9)
    obs_m = (preds["hg"] - preds["ag"]).to_numpy(dtype=float)
    exp_m = preds["exp_margin"].to_numpy(dtype=float)
    obs_t = (preds["hg"] + preds["ag"]).to_numpy(dtype=float)
    exp_t = preds["exp_total"].to_numpy(dtype=float)

    var_m = float(np.mean((obs_m - exp_m) ** 2))
    var_t = float(np.mean((obs_t - exp_t) ** 2))
    mean_tot = max(float(np.mean(exp_t)), 40.0)

    emp_fano = float(np.clip(var_m / mean_tot, 0.35, 1.35))
    emp_tempo_sd = float(np.clip(math.sqrt(max(var_t - var_m, 0.0)) / mean_tot, 0.01, 0.14))

    w = prior_weight
    disp_fano = (n * emp_fano + w * p0.disp_fano) / (n + w)
    tempo_sd_ratio = (n * emp_tempo_sd + w * p0.tempo_sd_ratio) / (n + w)
    return replace(p0, disp_fano=disp_fano, tempo_sd_ratio=tempo_sd_ratio)


def backtest_metrics(preds: pd.DataFrame) -> dict[str, float]:
    if preds.empty:
        return {}
    probs = preds[["p_home", "p_draw", "p_away"]].to_numpy()
    gd = (preds["hg"] - preds["ag"]).to_numpy()
    cls = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
    p_true = probs[np.arange(len(cls)), cls]
    resid_m = gd - preds["exp_margin"].to_numpy()
    resid_t = (preds["hg"] + preds["ag"]).to_numpy() - preds["exp_total"].to_numpy()
    return {
        "n_matches": float(len(preds)),
        "accuracy": float((probs.argmax(axis=1) == cls).mean()),
        "log_loss": float(-np.log(np.clip(p_true, 1e-12, None)).mean()),
        "brier": float(((probs - np.eye(3)[cls]) ** 2).sum(axis=1).mean()),
        "draw_rate_pred": float(probs[:, 1].mean()),
        "draw_rate_obs": float((cls == 1).mean()),
        "margin_mae": float(np.abs(resid_m).mean()),
        "margin_bias": float(resid_m.mean()),
        "total_mae": float(np.abs(resid_t).mean()),
        "total_bias": float(resid_t.mean()),
    }


# =========================================================================== #
# 8. Classement et départage officiel LNH
# =========================================================================== #
def _table_stats(home_idx, away_idx, hg, ag, n_teams: int):
    draw = (hg == ag).astype(int)
    home_pts = PTS_WIN * (hg > ag) + PTS_DRAW * draw
    away_pts = PTS_WIN * (ag > hg) + PTS_DRAW * draw
    pts = np.bincount(home_idx, home_pts, n_teams) + np.bincount(away_idx, away_pts, n_teams)
    gf = np.bincount(home_idx, hg, n_teams) + np.bincount(away_idx, ag, n_teams)
    ga = np.bincount(home_idx, ag, n_teams) + np.bincount(away_idx, hg, n_teams)
    return pts, gf, ga


def _break_tie(group: list, home_idx, away_idx, hg, ag, n_teams: int, gd_all, gf_all,
               reapply_h2h: bool, notes: Optional[dict]) -> list:
    in_group = np.zeros(n_teams, dtype=bool)
    in_group[group] = True
    mask = in_group[home_idx] & in_group[away_idx]
    pts_h, gf_h, ga_h = _table_stats(home_idx[mask], away_idx[mask], hg[mask], ag[mask], n_teams)
    key = {t: (-pts_h[t], -(gf_h[t] - ga_h[t]), -gf_h[t]) for t in group}

    if notes is not None:
        for c in range(3):
            if len({k[c] for k in key.values()}) > 1:
                for t in group:
                    notes[t] = H2H_LABELS[c]
                break

    result: list = []
    for _, grp in itertools.groupby(sorted(group, key=lambda t: key[t]), key=lambda t: key[t]):
        sub = list(grp)
        if len(sub) == 1:
            result.extend(sub)
        elif reapply_h2h and len(sub) < len(group):
            result.extend(_break_tie(sub, home_idx, away_idx, hg, ag, n_teams,
                                     gd_all, gf_all, reapply_h2h, notes))
        else:
            keys2 = {t: (-gd_all[t], -gf_all[t], t) for t in sub}
            if notes is not None:
                for c in range(3):
                    if len({k[c] for k in keys2.values()}) > 1:
                        for t in sub:
                            notes[t] = GENERAL_LABELS[c]
                        break
            result.extend(sorted(sub, key=lambda t: keys2[t]))
    return result


def rank_teams(home_idx, away_idx, hg, ag, n_teams: int, reapply_h2h: bool = True,
               notes: Optional[dict] = None) -> np.ndarray:
    home_idx, away_idx = np.asarray(home_idx), np.asarray(away_idx)
    hg, ag = np.asarray(hg), np.asarray(ag)
    pts, gf, ga = _table_stats(home_idx, away_idx, hg, ag, n_teams)
    gd = gf - ga
    base = np.argsort(-pts, kind="stable")
    out: list = []
    i = 0
    while i < n_teams:
        j = i
        while j < n_teams and pts[base[j]] == pts[base[i]]:
            j += 1
        group = base[i:j].tolist()
        if len(group) == 1:
            out.extend(group)
        else:
            out.extend(_break_tie(group, home_idx, away_idx, hg, ag, n_teams, gd, gf,
                                  reapply_h2h, notes))
        i = j
    return np.asarray(out, dtype=int)


def compute_standings(fx: pd.DataFrame, teams: list[str], reapply_h2h: bool = True) -> pd.DataFrame:
    n = len(teams)
    pl = fx[fx["played"]]
    if pl.empty:
        return pd.DataFrame({
            "Team": teams, "Played": 0, "W": 0, "D": 0, "L": 0,
            "GF": 0, "GA": 0, "GD": 0, "Real_Pts": 0, "Real_Rank": np.arange(1, n + 1),
        })
    h, a = pl["h_idx"].to_numpy(), pl["a_idx"].to_numpy()
    hg, ag = pl["home_ft_goals"].to_numpy(dtype=np.int64), pl["away_ft_goals"].to_numpy(dtype=np.int64)
    order = rank_teams(h, a, hg, ag, n, reapply_h2h)
    return _standings_frame(teams, h, a, hg, ag, order, "Real_Pts", "Real_Rank")


def _standings_frame(teams, h, a, hg, ag, order, pts_col: str, rank_col: str,
                     notes: Optional[dict] = None) -> pd.DataFrame:
    n = len(teams)
    pts, gf, ga = _table_stats(h, a, hg, ag, n)
    wins = np.bincount(h, hg > ag, n) + np.bincount(a, ag > hg, n)
    draws = np.bincount(h, hg == ag, n) + np.bincount(a, hg == ag, n)
    gp = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)
    df = pd.DataFrame({
        "Team": teams, "Played": gp, "W": wins.astype(int), "D": draws.astype(int),
        "L": (gp - wins - draws).astype(int), "GF": gf.astype(int), "GA": ga.astype(int),
        "GD": (gf - ga).astype(int), pts_col: pts.astype(int), rank_col: rank,
    })
    if notes is not None:
        df["Tiebreak"] = [notes.get(t, "") for t in range(n)]
    return df.sort_values(rank_col).reset_index(drop=True)


# =========================================================================== #
# 9. Stockage et export optimisé pour Datastorytelling (Parquet ZSTD + JSON)
# =========================================================================== #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    raise TypeError(f"Type non sérialisable JSON : {type(obj)}")


@dataclass
class SimulationStore:
    """Conserve les simulations en mémoire compacte et les exporte en Parquet ZSTD + JSON."""
    teams: list
    fixtures: pd.DataFrame
    hg: np.ndarray                # (S, M) int16
    ag: np.ndarray                # (S, M) int16
    points: np.ndarray            # (S, T) int16
    ranks: np.ndarray             # (S, T) int16
    exp_margin: np.ndarray        # (S, M) float32
    exp_total: np.ndarray         # (S, M) float32
    off_rtg_home: np.ndarray      # (S, M) float32
    def_rtg_home: np.ndarray      # (S, M) float32
    off_rtg_away: np.ndarray      # (S, M) float32
    def_rtg_away: np.ndarray      # (S, M) float32
    final_off_rtg: np.ndarray     # (S, T) float32
    final_def_rtg: np.ndarray     # (S, T) float32
    params: ModelParams
    cutoff_round: int
    season: str = ""

    @property
    def n_sims(self) -> int:
        return int(self.hg.shape[0])

    def team_index(self, name: str) -> int:
        q = _sort_key(canonical_team_name(name))
        exact = [i for i, t in enumerate(self.teams) if _sort_key(t) == q]
        if exact:
            return exact[0]
        cand = [i for i, t in enumerate(self.teams) if _sort_key(t).startswith(q)]
        cand = cand or [i for i, t in enumerate(self.teams) if q in _sort_key(t)]
        if len(cand) != 1:
            raise ValueError(f"Équipe '{name}' introuvable ou ambiguë. Équipes : {self.teams}")
        return cand[0]

    def extreme(self, team: str, best: bool = True) -> int:
        t = self.team_index(team)
        r, pts = self.ranks[:, t], self.points[:, t]
        target = r.min() if best else r.max()
        cand = np.flatnonzero(r == target)
        return int(cand[np.argmax(pts[cand]) if best else np.argmin(pts[cand])])

    def typical(self) -> int:
        mean = self.points.mean(axis=0)
        return int(np.argmin(((self.points - mean) ** 2).sum(axis=1)))

    def rank_distribution(self) -> pd.DataFrame:
        T = len(self.teams)
        probs = np.stack([
            (self.ranks == r).mean(axis=0)
            for r in range(1, T + 1)
        ], axis=1)

        df = pd.DataFrame(
            probs,
            index=self.teams,
            columns=[f"R{r}" for r in range(1, T + 1)]
        )

        order = np.argsort(self.ranks.mean(axis=0), kind="stable")
        return df.iloc[order]

    def _standings_chunk_df(self, start: int, stop: int) -> pd.DataFrame:
        """Construit les classements de [start:stop] de manière 100 % vectorisée et compacte."""
        S_c = stop - start
        M = len(self.fixtures)
        T = len(self.teams)
        H = self.fixtures["h_idx"].to_numpy()
        A = self.fixtures["a_idx"].to_numpy()

        oh_h = np.zeros((M, T), dtype=np.int16)
        oh_a = np.zeros((M, T), dtype=np.int16)
        oh_h[np.arange(M), H] = 1
        oh_a[np.arange(M), A] = 1
        gp_vec = (oh_h.sum(axis=0) + oh_a.sum(axis=0)).astype(np.int16)

        hg_c = self.hg[start:stop]
        ag_c = self.ag[start:stop]
        w_c = ((hg_c > ag_c).astype(np.int16) @ oh_h + (ag_c > hg_c).astype(np.int16) @ oh_a).astype(np.int16)
        d_c = ((hg_c == ag_c).astype(np.int16) @ (oh_h + oh_a)).astype(np.int16)
        l_c = (gp_vec[None, :] - w_c - d_c).astype(np.int16)
        gf_c = (hg_c.astype(np.int32) @ oh_h.astype(np.int32) + ag_c.astype(np.int32) @ oh_a.astype(np.int32)).astype(np.int16)
        ga_c = (ag_c.astype(np.int32) @ oh_h.astype(np.int32) + hg_c.astype(np.int32) @ oh_a.astype(np.int32)).astype(np.int16)
        gd_c = (gf_c - ga_c).astype(np.int16)

        sim_ids = np.repeat(np.arange(start, stop, dtype=np.int32), T)
        team_col = np.tile(np.asarray(self.teams, dtype=object), S_c)

        df = pd.DataFrame({
            "simulation": sim_ids,
            "Team": pd.Categorical(team_col, categories=self.teams),
            "Rank": self.ranks[start:stop].ravel().astype(np.int16),
            "Pts": self.points[start:stop].ravel().astype(np.int16),
            "Played": np.tile(gp_vec, S_c),
            "W": w_c.ravel(),
            "D": d_c.ravel(),
            "L": l_c.ravel(),
            "GF": gf_c.ravel(),
            "GA": ga_c.ravel(),
            "GD": gd_c.ravel(),
        })
        return df

    def _matches_chunk_df(self, start: int, stop: int) -> pd.DataFrame:
        """Construit les matchs de [start:stop] de manière 100 % vectorisée et compacte."""
        S_c = stop - start
        M = len(self.fixtures)
        fx = self.fixtures

        hg_c = self.hg[start:stop].ravel().astype(np.int16)
        ag_c = self.ag[start:stop].ravel().astype(np.int16)
        exp_m_c = self.exp_margin[start:stop].ravel().astype(np.float32)
        exp_t_c = self.exp_total[start:stop].ravel().astype(np.float32)
        exp_hg = 0.5 * (exp_t_c + exp_m_c)
        exp_ag = 0.5 * (exp_t_c - exp_m_c)
        p_h, p_d, p_a = outcome_probs_dc(exp_hg, exp_ag, self.params)

        res_arr = np.where(hg_c > ag_c, "1", np.where(hg_c == ag_c, "N", "2"))

        df = pd.DataFrame({
            "simulation": np.repeat(np.arange(start, stop, dtype=np.int32), M),
            "round": np.tile(fx["round"].to_numpy(dtype=np.int16), S_c),
            "match_id": pd.Categorical(np.tile(fx["match_id"].to_numpy(dtype=object), S_c)),
            "home_team": pd.Categorical(np.tile(fx["home_team"].to_numpy(dtype=object), S_c), categories=self.teams),
            "away_team": pd.Categorical(np.tile(fx["away_team"].to_numpy(dtype=object), S_c), categories=self.teams),
            "hg": hg_c,
            "ag": ag_c,
            "result": pd.Categorical(res_arr, categories=["1", "N", "2"]),
            "p_home": p_h.astype(np.float32),
            "p_draw": p_d.astype(np.float32),
            "p_away": p_a.astype(np.float32),
            "offRTG_h": self.off_rtg_home[start:stop].ravel().astype(np.float32),
            "defRTG_h": self.def_rtg_home[start:stop].ravel().astype(np.float32),
            "offRTG_a": self.off_rtg_away[start:stop].ravel().astype(np.float32),
            "defRTG_a": self.def_rtg_away[start:stop].ravel().astype(np.float32),
            "frozen": np.tile(fx["frozen"].to_numpy(dtype=bool), S_c),
        })
        return df

    def upcoming_scores(self) -> list[dict]:
        """Distribution exacte des scores pour tous les matchs simulés (après le cutoff ou non joués)."""
        fx, S, out = self.fixtures, self.n_sims, []
        has_real = "home_ft_goals" in fx.columns and "away_ft_goals" in fx.columns
        for i in np.flatnonzero(~fx["frozen"].to_numpy()):
            hg, ag = self.hg[:, i].astype(int), self.ag[:, i].astype(int)
            keys, cnt = np.unique(hg * 100 + ag, return_counts=True)
            scores = {f"{k // 100}-{k % 100}": int(c) for k, c in zip(keys, cnt)}
            top = int(np.argmax(cnt))
            real = (
                [int(fx["home_ft_goals"].iloc[i]), int(fx["away_ft_goals"].iloc[i])]
                if has_real and fx["played"].iloc[i]
                else None
            )
            out.append({
                "match_id": str(fx["match_id"].iloc[i]), "round": int(fx["round"].iloc[i]),
                "date": str(fx["date"].iloc[i]), "home": str(fx["home_team"].iloc[i]),
                "away": str(fx["away_team"].iloc[i]), "n": S,
                "p1": float((hg > ag).mean()), "pN": float((hg == ag).mean()), "p2": float((hg < ag).mean()),
                "exp_h": float(hg.mean()), "exp_a": float(ag.mean()),
                "modal": [int(keys[top] // 100), int(keys[top] % 100), float(cnt[top] / S)],
                "real": real,
                "scores": scores,
            })
        return out

    def export_storytelling(
        self,
        directory: str | Path = "output",
        pipeline_out: Optional[dict] = None,
        chunk_size: int = 500,
        include_matches: bool = True,
    ) -> dict[str, Path]:
        """Exporte toutes les données de simulation dans `directory` (Parquet ZSTD + JSON)
        en streaming par lots pour limiter l'empreinte RAM et disque."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        # 1. Résumé narratif JSON (léger, instantané pour web/visualisations)
        story_payload: dict[str, Any] = {
            "season": self.season,
            "cutoff_round": self.cutoff_round,
            "n_simulations": self.n_sims,
            "teams": self.teams,
            "params": asdict(self.params),
            "notable_simulations": {
                "typical_sim": self.typical(),
                "by_team": {
                    t: {
                        "best_sim": self.extreme(t, best=True),
                        "worst_sim": self.extreme(t, best=False),
                    }
                    for t in self.teams
                },
            },
            "rank_probabilities": self.rank_distribution().round(4).to_dict(orient="index"),
        }
        if pipeline_out is not None:
            if "team_aliases_applied" in pipeline_out:
                story_payload["team_aliases_applied"] = pipeline_out["team_aliases_applied"]
            story_payload["ratings_dc"] = pipeline_out["ratings_table"].round(3).to_dict(orient="records")
            story_payload["team_summary"] = pipeline_out["simulation"].round(4).to_dict(orient="records")
            if pipeline_out["fixtures"]["played"].any():
                story_payload["real_standings"] = pipeline_out["real_standings"].to_dict(orient="records")
                story_payload["comparison"] = pipeline_out["comparison"].round(3).to_dict(orient="records")
                story_payload["rank_metrics"] = pipeline_out["rank_metrics"]
            if pipeline_out["backtest_metrics"]:
                story_payload["backtest_metrics"] = pipeline_out["backtest_metrics"]

        json_path = d / "summary_storytelling.json"
        json_path.write_text(
            json.dumps(story_payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        written["summary_json"] = json_path

        up_path = d / "upcoming_scores.json"
        up_path.write_text(json.dumps(self.upcoming_scores(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        written["upcoming"] = up_path

        # 2. Tableau agrégé par équipe
        if pipeline_out is not None and "simulation" in pipeline_out:
            team_df = pipeline_out["simulation"].copy()
            if HAS_PARQUET:
                p_team = d / "team_summary.parquet"
                team_df.to_parquet(p_team, index=False, compression="zstd")
            else:
                p_team = d / "team_summary.csv.gz"
                team_df.to_csv(p_team, index=False, compression="gzip", encoding="utf-8")
            written["team_summary"] = p_team

        # 3. Classements de toutes les simulations (streaming par lots)
        if HAS_PARQUET:
            st_path = d / "sim_standings.parquet"
            writer_st = None
            try:
                for start in range(0, self.n_sims, chunk_size):
                    stop = min(start + chunk_size, self.n_sims)
                    df_chunk = self._standings_chunk_df(start, stop)
                    table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                    if writer_st is None:
                        writer_st = pq.ParquetWriter(st_path, table.schema, compression="zstd")
                    writer_st.write_table(table)
            finally:
                if writer_st is not None:
                    writer_st.close()
        else:
            st_path = d / "sim_standings.csv.gz"
            for start in range(0, self.n_sims, chunk_size):
                stop = min(start + chunk_size, self.n_sims)
                df_chunk = self._standings_chunk_df(start, stop)
                df_chunk.to_csv(
                    st_path,
                    mode="wt" if start == 0 else "at",
                    header=(start == 0),
                    index=False,
                    compression="gzip",
                    encoding="utf-8",
                )
        written["sim_standings"] = st_path

        # 4. Détail match par match de toutes les simulations (streaming par lots)
        if include_matches:
            if HAS_PARQUET:
                mx_path = d / "sim_matches.parquet"
                writer_mx = None
                try:
                    for start in range(0, self.n_sims, chunk_size):
                        stop = min(start + chunk_size, self.n_sims)
                        df_chunk = self._matches_chunk_df(start, stop)
                        table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                        if writer_mx is None:
                            writer_mx = pq.ParquetWriter(mx_path, table.schema, compression="zstd")
                        writer_mx.write_table(table)
                finally:
                    if writer_mx is not None:
                        writer_mx.close()
            else:
                mx_path = d / "sim_matches.csv.gz"
                for start in range(0, self.n_sims, chunk_size):
                    stop = min(start + chunk_size, self.n_sims)
                    df_chunk = self._matches_chunk_df(start, stop)
                    df_chunk.to_csv(
                        mx_path,
                        mode="wt" if start == 0 else "at",
                        header=(start == 0),
                        index=False,
                        compression="gzip",
                        encoding="utf-8",
                    )
            written["sim_matches"] = mx_path

        if not HAS_PARQUET:
            logger.info("pyarrow absent : export réalisé en CSV compressé (.csv.gz). "
                        "Installez 'pyarrow' (pip install pyarrow) pour obtenir du Parquet ZSTD encore plus compact.")
        return written


# =========================================================================== #
# 10. Moteur Monte Carlo
# =========================================================================== #
def simulate_season(
    fx: pd.DataFrame,
    dc: DixonColesFit,
    p: ModelParams,
    n_simulations: int = 5000,
    cutoff_round: int = 0,
    seed: Optional[int] = None,
    season: str = "",
) -> SimulationStore:
    rng = np.random.default_rng(seed)
    S, M, T = n_simulations, len(fx), len(dc.teams)
    H, A, R = fx["h_idx"].to_numpy(), fx["a_idx"].to_numpy(), fx["round"].to_numpy()
    frozen = fx["frozen"].to_numpy()
    HGr = fx["home_ft_goals"].fillna(0).to_numpy().astype(np.int64)
    AGr = fx["away_ft_goals"].fillna(0).to_numpy().astype(np.int64)
    RD = fx["rest_diff"].to_numpy()
    AOH, AOA = fx["avail_off_h"].to_numpy(), fx["avail_off_a"].to_numpy()
    ADH, ADA = fx["avail_def_h"].to_numpy(), fx["avail_def_a"].to_numpy()
    NH, NA = fx["n_prior_h"].to_numpy(), fx["n_prior_a"].to_numpy()

    state = RatingState.initial(S, dc)
    hg_all = np.zeros((S, M), dtype=np.int16)
    ag_all = np.zeros((S, M), dtype=np.int16)
    exp_m = np.zeros((S, M), dtype=np.float32)
    exp_t = np.zeros((S, M), dtype=np.float32)
    off_h_all = np.zeros((S, M), dtype=np.float32)
    def_h_all = np.zeros((S, M), dtype=np.float32)
    off_a_all = np.zeros((S, M), dtype=np.float32)
    def_a_all = np.zeros((S, M), dtype=np.float32)

    noise_scale = np.where(dc.is_promoted, p.rating_noise_sd * p.promoted_noise_mult, p.rating_noise_sd)
    noise_done = p.rating_noise_sd <= 0

    for i in range(M):
        if not noise_done and R[i] > cutoff_round:
            state.alpha += rng.normal(0.0, noise_scale[None, :], size=state.alpha.shape)
            state.beta += rng.normal(0.0, noise_scale[None, :], size=state.beta.shape)
            noise_done = True

        h, a = int(H[i]), int(A[i])
        lh, la = predict_dc(state, h, a, AOH[i], AOA[i], ADH[i], ADA[i], RD[i], dc, p)

        off_h_all[:, i] = np.exp(dc.mu_eff + state.alpha[:, h])
        def_h_all[:, i] = np.exp(dc.mu_eff + state.beta[:, h])
        off_a_all[:, i] = np.exp(dc.mu_eff + state.alpha[:, a])
        def_a_all[:, i] = np.exp(dc.mu_eff + state.beta[:, a])
        exp_m[:, i] = lh - la
        exp_t[:, i] = lh + la

        if frozen[i]:
            hg = np.full(S, HGr[i], dtype=np.int64)
            ag = np.full(S, AGr[i], dtype=np.int64)
        else:
            hg, ag = sample_scores_dc(rng, lh, la, p)

        update_state_dc(
            state, h, a, hg, ag, lh, la,
            int(NH[i]), int(NA[i]), bool(dc.is_promoted[h]), bool(dc.is_promoted[a]), p,
        )
        hg_all[:, i], ag_all[:, i] = hg, ag

    oh_h = np.zeros((M, T), dtype=np.float32)
    oh_a = np.zeros((M, T), dtype=np.float32)
    oh_h[np.arange(M), H] = 1.0
    oh_a[np.arange(M), A] = 1.0
    draw = (hg_all == ag_all).astype(np.float32)
    pts_h = PTS_WIN * (hg_all > ag_all).astype(np.float32) + PTS_DRAW * draw
    pts_a = PTS_WIN * (ag_all > hg_all).astype(np.float32) + PTS_DRAW * draw
    points = np.rint(pts_h @ oh_h + pts_a @ oh_a).astype(np.int16)

    ranks = np.empty((S, T), dtype=np.int16)
    positions = np.arange(1, T + 1)
    for s in range(S):
        order = rank_teams(H, A, hg_all[s].astype(np.int64), ag_all[s].astype(np.int64), T, p.reapply_h2h)
        ranks[s, order] = positions

    final_off = (np.exp(dc.mu_eff + state.alpha)).astype(np.float32)
    final_def = (np.exp(dc.mu_eff + state.beta)).astype(np.float32)

    fixtures = fx[[
        "match_id", "round", "date", "home_team", "away_team",
        "home_ft_goals", "away_ft_goals", "h_idx", "a_idx", "frozen", "played",
    ]].reset_index(drop=True).copy()

    return SimulationStore(
        teams=list(dc.teams), fixtures=fixtures, hg=hg_all, ag=ag_all,
        points=points, ranks=ranks, exp_margin=exp_m, exp_total=exp_t,
        off_rtg_home=off_h_all, def_rtg_home=def_h_all,
        off_rtg_away=off_a_all, def_rtg_away=def_a_all,
        final_off_rtg=final_off, final_def_rtg=final_def,
        params=p, cutoff_round=cutoff_round, season=season,
    )


# =========================================================================== #
# 11. Agrégation, comparaison et backtest multi-saisons
# =========================================================================== #
def aggregate_simulation(
    store: SimulationStore,
    n_relegated: int = 2,
    podium_size: int = 3,
    top_k: int = 4,
    pred_by: str = "mean_rank",
) -> tuple[pd.DataFrame, np.ndarray]:
    T = len(store.teams)
    rank_probs = np.stack([(store.ranks == r).mean(axis=0) for r in range(1, T + 1)], axis=1)
    q = np.percentile(store.points, [5, 50, 95], axis=0)
    df = pd.DataFrame({
        "Team": store.teams,
        "Mean_Rank": store.ranks.mean(axis=0),
        "Median_Rank": np.median(store.ranks, axis=0),
        "Std_Rank": store.ranks.std(axis=0),
        "P_1st": (store.ranks == 1).mean(axis=0),
        "P_Podium": (store.ranks <= podium_size).mean(axis=0),
        f"P_Top{top_k}": (store.ranks <= top_k).mean(axis=0),
        "P_Relegated": (store.ranks > T - n_relegated).mean(axis=0),
        "Expected_Pts": store.points.mean(axis=0),
        "Pts_P05": q[0], "Pts_P50": q[1], "Pts_P95": q[2],
        "Final_offRTG": store.final_off_rtg.mean(axis=0),
        "Final_defRTG": store.final_def_rtg.mean(axis=0),
    })
    if pred_by == "expected_points":
        keys, asc = ["Expected_Pts", "Mean_Rank", "Team"], [False, True, True]
    else:
        keys, asc = ["Mean_Rank", "Expected_Pts", "Team"], [True, False, True]
    order = df.sort_values(keys, ascending=asc).index
    df["Pred_Rank"] = 0
    df.loc[order, "Pred_Rank"] = np.arange(1, T + 1)
    return df.sort_values("Pred_Rank").reset_index(drop=True), rank_probs


def build_comparison(real: pd.DataFrame, agg: pd.DataFrame, rank_probs: np.ndarray,
                     teams: list[str], fx: pd.DataFrame) -> pd.DataFrame:
    games = pd.concat([fx["home_team"], fx["away_team"]]).value_counts()
    m = real[["Team", "Real_Rank", "Real_Pts", "Played"]].merge(
        agg[["Team", "Pred_Rank", "Expected_Pts", "Mean_Rank", "Pts_P05", "Pts_P95"]], on="Team"
    )
    m["Rank_Diff"] = m["Pred_Rank"] - m["Real_Rank"]
    m["Pts_Diff"] = m["Real_Pts"] - m["Expected_Pts"]
    m["Real_Played"] = m["Played"]
    m["Sim_Games"] = m["Team"].map(games)
    m["P_Real_Rank"] = [rank_probs[teams.index(t), int(r) - 1] for t, r in zip(m["Team"], m["Real_Rank"])]
    m["In_90_CI"] = (m["Real_Pts"] >= m["Pts_P05"]) & (m["Real_Pts"] <= m["Pts_P95"])
    cols = [
        "Team", "Real_Rank", "Pred_Rank", "Rank_Diff", "Real_Pts", "Expected_Pts", "Pts_Diff",
        "Mean_Rank", "P_Real_Rank", "In_90_CI", "Real_Played", "Sim_Games",
    ]
    return m[cols].sort_values("Real_Rank").reset_index(drop=True)


def rank_metrics(comp: pd.DataFrame) -> dict[str, float]:
    d = comp["Rank_Diff"].to_numpy(dtype=float)
    r, pr = comp["Real_Rank"].to_numpy(dtype=float), comp["Pred_Rank"].to_numpy(dtype=float)
    spearman = float(np.corrcoef(r, pr)[0, 1]) if len(comp) > 1 else float("nan")
    return {
        "MAE_rank": float(np.abs(d).mean()),
        "RMSE_rank": float(np.sqrt((d ** 2).mean())),
        "Spearman": spearman,
        "Exact_rank_share": float((d == 0).mean()),
        "MAE_points": float(comp["Pts_Diff"].abs().mean()),
        "Mean_P_Real_Rank": float(comp["P_Real_Rank"].mean()),
        "CI90_coverage": float(comp["In_90_CI"].mean()),
    }


def _fmt(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x:.{digits}f}")


# =========================================================================== #
# 12. Orchestration et CLI
# =========================================================================== #
def run_pipeline(
    match_csv: str | Path,
    boxscore_csv: Optional[str | Path] = None,
    players_csv: Optional[str | Path] = None,
    season: Optional[str] = None,
    n_simulations: int = 5000,
    cutoff_round: int = 0,
    seed: Optional[int] = 42,
    params: Optional[ModelParams] = None,
    fill_missing: bool = False,
    n_relegated: int = 2,
    top_k: int = 4,
    pred_by: str = "mean_rank",
    _preloaded_all: Optional[tuple[pd.DataFrame, list[str], Optional[pd.DataFrame], dict[str, str]]] = None,
) -> dict:
    p0 = params or ModelParams()

    if _preloaded_all is not None:
        all_matches, seasons, box, aliases_applied = _preloaded_all
    else:
        alias_map = load_team_aliases(match_csv)
        has_box = bool(boxscore_csv and _find_csvs(boxscore_csv, "players_boxscore.csv"))
        box = load_player_tables(boxscore_csv, players_csv, alias_map) if has_box else None
        all_matches, seasons, aliases_applied = load_all_matches(
            match_csv, box, alias_map=alias_map
        )

    if season is None:
        season = seasons[-1]
        logger.info("Saison non précisée -> dernière saison trouvée : %s", season)
    elif season not in seasons:
        alt = season.replace("-", "/") if "-" in season else season.replace("/", "-")
        if alt in seasons:
            season = alt

    matches = all_matches[all_matches["season"] == season].copy()
    if matches.empty:
        raise ValueError(f"Aucun match pour la saison '{season}'. Saisons disponibles : {seasons}")

    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]), key=_sort_key)
    check_season_completeness(matches, teams)
    if fill_missing:
        matches = fill_missing_fixtures(matches, teams, seed or 0)
    fx = prepare_fixtures(matches, teams, cutoff_round, p0)

    s_idx = int(matches["season_idx"].iloc[0])
    prev_matches = all_matches[all_matches["season_idx"] == (s_idx - 1)]
    roster_adj = compute_roster_turnover(box, prev_matches, teams, p0) if cutoff_round == 0 else {}

    dc = fit_dixon_coles_ratings(all_matches, season, teams, cutoff_round, p0, roster_adj)
    p = fit_dispersion_params(fx, dc, p0, cutoff_round)

    known = fx[fx["frozen"]]
    pen_off, pen_def = compute_availability(box, known, teams, p)
    apply_availability(fx, pen_off, pen_def, cutoff_round, p)

    _, _, preds = run_history(fx, dc, p, cutoff_round)
    real = compute_standings(fx, teams, p.reapply_h2h)

    store = simulate_season(fx, dc, p, n_simulations, cutoff_round, seed, season)
    agg, rank_probs = aggregate_simulation(store, min(n_relegated, len(teams) - 1), 3, top_k, pred_by)
    comp = build_comparison(real, agg, rank_probs, teams, fx)

    return {
        "season": season,
        "available_seasons": seasons,
        "teams": teams,
        "team_aliases_applied": aliases_applied,
        "params": p,
        "dc_fit": dc,
        "ratings_table": dc.summary_table(),
        "fixtures": fx,
        "store": store,
        "availability": pd.DataFrame({"off_log_pen": pen_off, "def_log_pen": pen_def}, index=teams),
        "backtest_predictions": preds,
        "backtest_metrics": backtest_metrics(preds),
        "real_standings": real,
        "simulation": agg,
        "comparison": comp,
        "rank_metrics": rank_metrics(comp),
    }


def run_multi_season_backtest(
    match_csv: str | Path,
    boxscore_csv: Optional[str | Path] = None,
    players_csv: Optional[str | Path] = None,
    cutoff_round: int = 0,
    n_simulations: int = 2000,
    seed: int = 42,
    params: Optional[ModelParams] = None,
    min_prior_seasons: int = 1,
) -> pd.DataFrame:
    p0 = params or ModelParams()
    alias_map = load_team_aliases(match_csv)
    has_box = bool(boxscore_csv and _find_csvs(boxscore_csv, "players_boxscore.csv"))
    box = load_player_tables(boxscore_csv, players_csv, alias_map) if has_box else None
    all_matches, seasons, aliases_applied = load_all_matches(
        match_csv, box, alias_map=alias_map
    )
    if len(seasons) <= min_prior_seasons:
        raise ValueError(f"Pas assez de saisons ({len(seasons)}) pour un backtest multi-saisons.")

    summary_rows = []
    for s_idx in range(min_prior_seasons, len(seasons)):
        s = seasons[s_idx]
        s_matches = all_matches[all_matches["season"] == s]
        if s_matches["played"].sum() < 20:
            continue
        out = run_pipeline(
            match_csv, boxscore_csv, players_csv,
            season=s, n_simulations=n_simulations, cutoff_round=cutoff_round,
            seed=seed + s_idx, params=p0,
            _preloaded_all=(all_matches, seasons, box, aliases_applied),
        )
        bm = out["backtest_metrics"]
        rm = out["rank_metrics"]
        summary_rows.append({
            "Season": s,
            "Teams": len(out["teams"]),
            "Promoted": int(out["dc_fit"].is_promoted.sum()),
            "Accuracy": bm.get("accuracy", np.nan),
            "LogLoss": bm.get("log_loss", np.nan),
            "Brier": bm.get("brier", np.nan),
            "Margin_MAE": bm.get("margin_mae", np.nan),
            "MAE_Rank": rm.get("MAE_rank", np.nan),
            "Spearman": rm.get("Spearman", np.nan),
            "MAE_Pts": rm.get("MAE_points", np.nan),
            "CI90_Cov": rm.get("CI90_coverage", np.nan),
        })
    return pd.DataFrame(summary_rows)


def _print_report(out: dict, cutoff_round: int) -> None:
    p: ModelParams = out["params"]
    dc: DixonColesFit = out["dc_fit"]
    store: SimulationStore = out["store"]
    n_frozen = int(out["fixtures"]["frozen"].sum())
    mean_goals_team = math.exp(dc.mu_eff)
    mean_goals_match = 2.0 * mean_goals_team

    print(f"\n=== LNH Starligue {out['season']} (Dixon & Coles Handball) | {len(out['teams'])} équipes "
          f"| {store.n_sims} simulations | cutoff_round={cutoff_round} ===")
    print(f"Historique exploité : {len(out['available_seasons'])} saisons ({out['available_seasons'][0]} -> {out['season']})")
    if out.get("team_aliases_applied"):
        print(f"Noms de clubs unifiés : {len(out['team_aliases_applied'])} variantes fusionnées")
    print(f"Matchs réels figés : {n_frozen} | matchs simulés par itération : {len(out['fixtures']) - n_frozen}")
    print(f"Paramètres : {mean_goals_match:.1f} buts/match, Fano={p.disp_fano:.3f}, tempo sd={p.tempo_sd_ratio:.3f}, "
          f"xi={p.dc_xi:.2f}/an, rétention={p.season_reversion:.2f}")

    print("\n--- Ratings Dixon & Coles au cutoff (buts marqués / encaissés par match) ---")
    print(_fmt(out["ratings_table"], 2))

    if out["backtest_metrics"]:
        print("\n--- Backtest pas-à-pas (matchs joués de la saison) ---")
        for k, v in out["backtest_metrics"].items():
            print(f"{k:>16s} : {int(v) if k == 'n_matches' else round(v, 3)}")

    print("\n--- Prédictions Monte Carlo ---")
    print(_fmt(out["simulation"], 3))

    if out["fixtures"]["played"].any():
        print("\n--- Comparaison rang réel / rang prédit ---")
        print(_fmt(out["comparison"], 2))
        print("\n--- Métriques globales de classement ---")
        for k, v in out["rank_metrics"].items():
            print(f"{k:>18s} : {v:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulateur LNH Starligue Dixon & Coles (Export Datastorytelling)")
    ap.add_argument("--data-dir", default=".", help="Dossier contenant les sous-dossiers d'années ou les CSV")
    ap.add_argument("--output-dir", default="output", help="Dossier d'export par défaut (défaut: ./output)")
    ap.add_argument("--season", default=None, help="Saison cible (ex. 2026-2027)")
    ap.add_argument("--n-simulations", type=int, default=5000)
    ap.add_argument("--cutoff-round", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fill-missing", action="store_true", help="Génère les matchs absents du CSV")
    ap.add_argument("--no-players", action="store_true", help="Ignore les données joueurs")
    ap.add_argument("--no-export-matches", action="store_true",
                    help="N'exporte pas sim_matches.parquet (les scores à venir restent dans upcoming_scores.json)")
    ap.add_argument("--chunk-size", type=int, default=500, help="Taille des lots d'écriture Parquet (défaut: 500)")
    ap.add_argument("--n-relegated", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--pred-by", choices=["mean_rank", "expected_points"], default="mean_rank")
    ap.add_argument("--param", action="append", default=[], metavar="NOM=VAL",
                    help="Surcharge un ModelParams, ex. --param dc_xi=0.75 (répétable)")
    ap.add_argument("--backtest-seasons", action="store_true",
                    help="Lance un backtest walk-forward sur toutes les saisons historiques")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    d = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    params = replace(ModelParams(), **parse_param_overrides(args.param))

    if args.backtest_seasons:
        bt = run_multi_season_backtest(
            d,
            None if args.no_players else d,
            None if args.no_players else d,
            cutoff_round=args.cutoff_round,
            n_simulations=args.n_simulations,
            seed=args.seed,
            params=params,
        )
        print("\n=== Backtest Walk-Forward Multi-Saisons (Dixon & Coles Handball) ===")
        print(_fmt(bt, 3))
        if not bt.empty:
            out_dir.mkdir(parents=True, exist_ok=True)
            bt_path = out_dir / "backtest_multi_seasons.csv"
            bt.to_csv(bt_path, index=False, encoding="utf-8")
            print(f"\nRésultats du backtest enregistrés dans : {bt_path}")
        return

    out = run_pipeline(
        d,
        None if args.no_players else d,
        None if args.no_players else d,
        args.season, args.n_simulations, args.cutoff_round, args.seed, params,
        args.fill_missing, args.n_relegated, args.top_k, args.pred_by,
    )
    _print_report(out, args.cutoff_round)

    store: SimulationStore = out["store"]
    written = store.export_storytelling(
        directory=out_dir,
        pipeline_out=out,
        chunk_size=args.chunk_size,
        include_matches=not args.no_export_matches,
    )
    print(f"\n=== Fichiers datastorytelling enregistrés dans '{out_dir}/' ===")
    for label, fpath in written.items():
        size_mb = fpath.stat().st_size / (1024 * 1024)
        print(f"  - {label:<15s} : {fpath.name} ({size_mb:.2f} Mo)")


if __name__ == "__main__":
    main()