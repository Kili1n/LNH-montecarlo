import argparse
import csv
import html
import re
import unicodedata
from pathlib import Path
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTChar
from tqdm import tqdm

# CONFIG
ANNEE = "2026-2027"
RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_PDFS_RACINE = RACINE_PROJET / "data/feuilles_de_match"
DOSSIER_CSV_RACINE = RACINE_PROJET / "data/data_csv"

BASE_MATCH_COLS = [
    "match_id", "season", "round", "date", "time", "venue",
    "home_team", "away_team", "home_ht_goals", "away_ht_goals",
    "home_ft_goals", "away_ft_goals",
    "home_goals", "home_shots", "home_goals_7m", "home_shots_7m",
    "home_saves", "home_shots_faced", "home_saves_7m", "home_shots_faced_7m",
    "home_br", "home_tc", "home_pp", "home_2p", "home_bp", "home_dp",
    "home_a", "home_2min", "home_d", "home_lnh_score",
    "away_goals", "away_shots", "away_goals_7m", "away_shots_7m",
    "away_saves", "away_shots_faced", "away_saves_7m", "away_shots_faced_7m",
    "away_br", "away_tc", "away_pp", "away_2p", "away_bp", "away_dp",
    "away_a", "away_2min", "away_d", "away_lnh_score",
]

# Table d'équivalence pour reconnaître qu'un match futur/sans PDF et un PDF concernent le même club
ALIAS_EQUIPES: dict[str, str] = {
    "psg": "psg", "psg handball": "psg", "paris": "psg", "paris sg": "psg",
    "paris saint germain": "psg", "paris saint germain handball": "psg",
    "paris hb": "psg", "paris handball": "psg",
    "hbc nantes": "nantes", "nantes": "nantes", "handball club de nantes": "nantes",
    "handball club nantes": "nantes",
    "montpellier hb": "montpellier", "montpellier": "montpellier",
    "montpellier handball": "montpellier", "mhb": "montpellier",
    "montpellier agglomeration handball": "montpellier", "mahb": "montpellier",
    "fenix toulouse": "toulouse", "toulouse": "toulouse",
    "fenix toulouse handball": "toulouse", "toulouse handball": "toulouse", "fth": "toulouse",
    "chambery": "chambery", "chambery savoie": "chambery",
    "chambery savoie handball": "chambery", "chambery savoie mont blanc handball": "chambery",
    "chambery smbh": "chambery", "csmbh": "chambery",
    "usam nimes": "nimes", "nimes": "nimes", "usam nimes gard": "nimes", "usam": "nimes",
    "saint raphael": "saint_raphael", "st raphael": "saint_raphael",
    "saint raphael var handball": "saint_raphael", "st raphael var handball": "saint_raphael",
    "srvhb": "saint_raphael",
    "pauc": "pauc", "pauc handball": "pauc", "aix": "pauc", "aix en provence": "pauc",
    "pays d aix": "pauc", "pays d aix uc": "pauc", "pays d aix universite club": "pauc",
    "pays d aix universite club handball": "pauc",
    "limoges": "limoges", "limoges handball": "limoges", "limoges hand 87": "limoges",
    "lh 87": "limoges", "lh87": "limoges",
    "usdk": "dunkerque", "dunkerque": "dunkerque", "usdk dunkerque": "dunkerque",
    "dunkerque handball": "dunkerque", "dunkerque handball grand littoral": "dunkerque",
    "dunkerque hgl": "dunkerque",
    "cesson rennes": "cesson", "cesson": "cesson", "cesson rennes metropole": "cesson",
    "cesson rennes metropole handball": "cesson", "cesson rennes mhb": "cesson", "crmhb": "cesson",
    "ivry": "ivry", "us ivry": "ivry", "us ivry handball": "ivry",
    "creteil": "creteil", "us creteil": "creteil", "us creteil handball": "creteil",
    "chartres": "chartres", "c chartres": "chartres",
    "c chartres metropole handball": "chartres", "chartres metropole handball": "chartres",
    "chartres mhb": "chartres", "ccmh": "chartres",
    "istres": "istres", "istres provence": "istres",
    "istres provence handball": "istres", "istres ouest provence handball": "istres", "ioph": "istres",
    "tremblay": "tremblay", "tremblay en france": "tremblay",
    "tremblay handball": "tremblay", "tremblay en france handball": "tremblay", "tfhb": "tremblay",
    "saran": "saran", "saran loiret": "saran", "saran loiret handball": "saran", "usm saran": "saran",
    "selestat": "selestat", "selestat alsace": "selestat",
    "selestat alsace handball": "selestat", "sahb": "selestat",
    "dijon": "dijon", "dijon metropole": "dijon",
    "dijon metropole handball": "dijon", "dijon bourgogne handball": "dijon", "dbhb": "dijon",
    "nancy": "nancy", "grand nancy": "nancy",
    "grand nancy metropole": "nancy", "grand nancy metropole handball": "nancy", "gnmhb": "nancy",
    "massy": "massy", "massy essonne": "massy", "massy essonne handball": "massy", "mehb": "massy",
    "pontault combault": "pontault", "pontault": "pontault",
    "ums pontault combault": "pontault", "ums pontault combault handball": "pontault",
    "billere": "billere", "billere handball": "billere", "billere pau pyrenees": "billere",
    "caen": "caen", "caen handball": "caen", "caen hb": "caen",
    "frontignan": "frontignan", "frontignan thau": "frontignan",
    "frontignan thau handball": "frontignan", "fthb": "frontignan",
    "cherbourg": "cherbourg", "js cherbourg": "cherbourg", "js cherbourg manche handball": "cherbourg",
    "sarrebourg": "sarrebourg", "sarrebourg moselle sud handball": "sarrebourg", "smshb": "sarrebourg",
    "valence": "valence", "valence handball": "valence",
    "besancon": "besancon", "grand besancon": "besancon",
    "grand besancon doubs handball": "besancon", "gbdh": "besancon",
}

# Coordonnées horizontales (x_min, x_max) fixes des colonnes LNH
COLS = {
    "jersey": (0, 21),
    "last_name": (21, 110),
    "first_name": (110, 180),
    "time": (180, 204),
    "buts_champ": (204, 226),
    "buts_7m": (226, 248),
    "total_buts": (248, 270),
    "arrets": (291, 314),
    "arrets_7m": (334, 359),
    "br": (381, 404),
    "tc": (404, 427),
    "pp": (427, 451),
    "2p": (451, 473),
    "bp": (473, 496),
    "dp": (496, 519),
    "a": (519, 533),
    "2min": (533, 549),
    "d": (549, 564),
    "score_lnh": (564, 600),
}

HAUTEUR_PAGE_VIRTUELLE = 1000.0


def _normaliser_texte(nom: str) -> str:
    s = unicodedata.normalize("NFKD", str(nom).strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def _cle_equipe(nom: str) -> str:
    """Normalise le nom d'une équipe pour éviter tout doublon entre CSV annexes et PDF."""
    cle = _normaliser_texte(nom)
    return ALIAS_EQUIPES.get(cle, cle)


def extract_rows(pdf_path: Path) -> list[dict]:
    """
    Extrait tous les caractères sur TOUTES les pages du PDF (indispensable quand un match
    déborde sur 2+ pages à cause d'un grand nombre de joueurs ou du bug des 30 coachs).
    """
    chars = []

    def walk(obj, page_idx: int):
        if isinstance(obj, LTChar):
            y_global = obj.bbox[1] - (page_idx * HAUTEUR_PAGE_VIRTUELLE)
            chars.append((page_idx, y_global, obj.bbox[0], obj.bbox[2], obj.get_text()))
        elif hasattr(obj, "__iter__"):
            for child in obj:
                walk(child, page_idx)

    for page_idx, page_layout in enumerate(extract_pages(str(pdf_path))):
        walk(page_layout, page_idx)

    chars.sort(key=lambda c: (c[0], -round(c[1], 1), c[2]))
    rows = []
    for page_idx, y_global, x0, x1, txt in chars:
        if (
            not rows
            or rows[-1]["page"] != page_idx
            or abs(y_global - rows[-1]["y"]) > 2.0
        ):
            rows.append({"page": page_idx, "y": y_global, "chars": [(x0, x1, txt)]})
        else:
            rows[-1]["chars"].append((x0, x1, txt))
    return rows


def cell_text(row_chars: list[tuple[float, float, str]], x_min: float, x_max: float) -> str:
    """Reconstruit le texte d'une cellule dans l'intervalle horizontal [x_min, x_max)."""
    selected = [(x0, x1, txt) for x0, x1, txt in row_chars if x_min <= x0 < x_max]
    if not selected:
        return ""
    selected.sort(key=lambda c: c[0])
    out = []
    prev_x1 = None
    for x0, x1, txt in selected:
        if prev_x1 is not None and (x0 - prev_x1) > 1.8:
            out.append(" ")
        out.append(txt)
        prev_x1 = x1
    return html.unescape("".join(out)).strip()


def parse_ratio(val: str) -> tuple[int, int]:
    """Sépare 'X / Y' en (X, Y). Renvoie (0, 0) si la case est vide ou vaut '0'."""
    val = val.strip()
    if not val or val == "0":
        return 0, 0
    m = re.match(r"^(\d+)\s*/\s*(\d+)$", val)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def parse_int(val: str) -> int:
    val = val.strip()
    if not val:
        return 0
    m = re.search(r"-?\d+", val)
    return int(m.group(0)) if m else 0


def parse_float(val: str) -> float:
    val = val.strip().replace(",", ".")
    if not val:
        return 0.0
    m = re.search(r"-?\d+(?:\.\d+)?", val)
    return float(m.group(0)) if m else 0.0


def mmss_to_seconds(val: str) -> int:
    """Convertit 'MM:SS' en nombre total de secondes."""
    m = re.match(r"^(\d+):(\d{2})$", val.strip())
    return int(m.group(1)) * 60 + int(m.group(2)) if m else 0


def date_fr_to_iso(val: str) -> str:
    """Convertit 'JJ/MM/AAAA' en 'AAAA-MM-JJ'."""
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", val.strip())
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else val.strip()


def _est_titre_equipe_attendu(titre: str, header_home: str, header_away: str) -> int | None:
    """
    Vérifie si 'titre' correspond à l'équipe domicile (retourne 0) ou extérieur (retourne 1).
    Utilise une correspondance exacte puis normalisée pour éviter tout faux positif.
    """
    t_low = titre.strip().lower()
    if not t_low:
        return None
    if header_home and t_low == header_home.strip().lower():
        return 0
    if header_away and t_low == header_away.strip().lower():
        return 1

    t_norm = _normaliser_texte(titre)
    if header_home and t_norm == _normaliser_texte(header_home):
        return 0
    if header_away and t_norm == _normaliser_texte(header_away):
        return 1
    return None


def parse_lnh_pdf(pdf_path: Path) -> tuple[dict, list[dict], bool]:
    """
    Parse une feuille de match LNH (mono-page ou multi-pages, ou feuille vide avec score global).
    Retourne (match_row, player_rows, feuille_detaillee).
    """
    rows = extract_rows(pdf_path)

    meta = {}
    home_ht = away_ht = home_ft = away_ft = 0
    has_ft = False
    header_home_name = ""
    header_away_name = ""

    # 1. En-tête : métadonnées et scores (page 0, y > 700)
    for r in rows:
        if r["page"] != 0 or r["y"] < 700:
            continue
        y = r["y"]

        left_label = cell_text(r["chars"], 100, 185)
        left_val = cell_text(r["chars"], 185, 345)
        right_label = cell_text(r["chars"], 345, 430)
        right_val = cell_text(r["chars"], 430, 580)

        if left_label == "Code match":
            meta["match_id"] = left_val
        elif left_label == "Date":
            meta["date"] = date_fr_to_iso(left_val)
        elif left_label == "Lieu":
            meta["venue"] = left_val

        if right_label == "Saison":
            meta["season"] = re.sub(r"\s*/\s*", "/", right_val)
        elif right_label == "Journée":
            m_round = re.search(r"Journée\s*0*(\d+)", right_val)
            meta["round"] = int(m_round.group(1)) if m_round else right_val
        elif right_label == "Heure":
            meta["time"] = right_val

        if 740 <= y <= 765:
            h_txt = cell_text(r["chars"], 50, 295)
            a_txt = cell_text(r["chars"], 300, 595)
            if h_txt and a_txt:
                header_home_name = h_txt
                header_away_name = a_txt

        center_lbl = cell_text(r["chars"], 250, 345)
        if center_lbl == "Mi-temps":
            home_ht = parse_int(cell_text(r["chars"], 80, 200))
            away_ht = parse_int(cell_text(r["chars"], 400, 520))
        elif center_lbl == "Score final":
            h_ft_txt = cell_text(r["chars"], 80, 200)
            a_ft_txt = cell_text(r["chars"], 400, 520)
            if h_ft_txt or a_ft_txt:
                has_ft = True
            home_ft = parse_int(h_ft_txt)
            away_ft = parse_int(a_ft_txt)

    # 2. Tableaux statistiques domicile puis extérieur sur TOUTES les pages
    teams_slots: list[dict | None] = [None, None]
    current_team: dict | None = None
    equipes_deja_vues: set[int] = set()

    for r in rows:
        if r["page"] == 0 and r["y"] > 700:
            continue

        full_line = cell_text(r["chars"], 0, 600)
        if full_line.startswith("* B.R."):
            break

        jersey_txt = cell_text(r["chars"], *COLS["jersey"])
        last_name_txt = cell_text(r["chars"], *COLS["last_name"])
        team_title_txt = cell_text(r["chars"], 21, 180)
        stats_area_txt = cell_text(r["chars"], 180, 600)

        # Détection uniquement des 2 équipes du match (ignore les officiels/coachs même s'il y en a 30)
        if not jersey_txt and not stats_area_txt and team_title_txt:
            slot_idx = _est_titre_equipe_attendu(team_title_txt, header_home_name, header_away_name)
            if slot_idx is not None and slot_idx not in equipes_deja_vues:
                current_team = {
                    "name": team_title_txt,
                    "players": [],
                    "totals": {},
                }
                teams_slots[slot_idx] = current_team
                equipes_deja_vues.add(slot_idx)
                continue

        if current_team is None:
            continue

        # Ligne des totaux d'équipe ("16 JOUEURS", "15 JOUEURS", etc.)
        if re.search(r"\d+\s+JOUEURS", last_name_txt):
            goals, shots = parse_ratio(cell_text(r["chars"], *COLS["total_buts"]))
            goals_7m, shots_7m = parse_ratio(cell_text(r["chars"], *COLS["buts_7m"]))
            saves, shots_faced = parse_ratio(cell_text(r["chars"], *COLS["arrets"]))
            saves_7m, shots_faced_7m = parse_ratio(cell_text(r["chars"], *COLS["arrets_7m"]))

            current_team["totals"] = {
                "goals": goals,
                "shots": shots,
                "goals_7m": goals_7m,
                "shots_7m": shots_7m,
                "saves": saves,
                "shots_faced": shots_faced,
                "saves_7m": saves_7m,
                "shots_faced_7m": shots_faced_7m,
                "br": parse_int(cell_text(r["chars"], *COLS["br"])),
                "tc": parse_int(cell_text(r["chars"], *COLS["tc"])),
                "pp": parse_int(cell_text(r["chars"], *COLS["pp"])),
                "2p": parse_int(cell_text(r["chars"], *COLS["2p"])),
                "bp": parse_int(cell_text(r["chars"], *COLS["bp"])),
                "dp": parse_int(cell_text(r["chars"], *COLS["dp"])),
                "a": parse_int(cell_text(r["chars"], *COLS["a"])),
                "2min": parse_int(cell_text(r["chars"], *COLS["2min"])),
                "d": parse_int(cell_text(r["chars"], *COLS["d"])),
                "lnh_score": parse_float(cell_text(r["chars"], *COLS["score_lnh"])),
            }
            # On ferme le bloc équipe après sa ligne de totaux afin d'ignorer
            # les 30+ lignes d'officiels/coachs qui suivent dans le bug LNH 25/26
            current_team = None
            continue

        # Ligne joueur individuel (uniquement numéro numérique, les officiels A/B/C/D sont ignorés)
        if jersey_txt.isdigit():
            goals, shots = parse_ratio(cell_text(r["chars"], *COLS["total_buts"]))
            goals_7m, shots_7m = parse_ratio(cell_text(r["chars"], *COLS["buts_7m"]))
            saves, shots_faced = parse_ratio(cell_text(r["chars"], *COLS["arrets"]))
            saves_7m, shots_faced_7m = parse_ratio(cell_text(r["chars"], *COLS["arrets_7m"]))

            current_team["players"].append({
                "jersey_num": int(jersey_txt),
                "last_name": last_name_txt,
                "first_name": cell_text(r["chars"], *COLS["first_name"]),
                "seconds_played": mmss_to_seconds(cell_text(r["chars"], *COLS["time"])),
                "goals": goals,
                "shots": shots,
                "goals_7m": goals_7m,
                "shots_7m": shots_7m,
                "saves": saves,
                "shots_faced": shots_faced,
                "saves_7m": saves_7m,
                "shots_faced_7m": shots_faced_7m,
                "br": parse_int(cell_text(r["chars"], *COLS["br"])),
                "tc": parse_int(cell_text(r["chars"], *COLS["tc"])),
                "pp": parse_int(cell_text(r["chars"], *COLS["pp"])),
                "2p": parse_int(cell_text(r["chars"], *COLS["2p"])),
                "bp": parse_int(cell_text(r["chars"], *COLS["bp"])),
                "dp": parse_int(cell_text(r["chars"], *COLS["dp"])),
                "a": parse_int(cell_text(r["chars"], *COLS["a"])),
                "2min": parse_int(cell_text(r["chars"], *COLS["2min"])),
                "d": parse_int(cell_text(r["chars"], *COLS["d"])),
                "lnh_score": parse_float(cell_text(r["chars"], *COLS["score_lnh"])),
            })

    match_id = meta.get("match_id", pdf_path.stem)
    home_info = teams_slots[0]
    away_info = teams_slots[1]

    # Cas "Feuille de match vide (seul score global)" ex: 12/13 j15_cesson-rennes_selestat.pdf
    if home_info is None or away_info is None:
        if header_home_name and header_away_name and has_ft:
            match_row = {
                "match_id": match_id,
                "season": meta.get("season", ""),
                "round": meta.get("round", ""),
                "date": meta.get("date", ""),
                "time": meta.get("time", ""),
                "venue": meta.get("venue", ""),
                "home_team": header_home_name,
                "away_team": header_away_name,
                "home_ht_goals": home_ht,
                "away_ht_goals": away_ht,
                "home_ft_goals": home_ft,
                "away_ft_goals": away_ft,
            }
            return match_row, [], False
        raise ValueError(f"Structure incomplète dans {pdf_path.name}")

    home_name = home_info["name"] or header_home_name
    away_name = away_info["name"] or header_away_name

    match_row = {
        "match_id": match_id,
        "season": meta.get("season", ""),
        "round": meta.get("round", ""),
        "date": meta.get("date", ""),
        "time": meta.get("time", ""),
        "venue": meta.get("venue", ""),
        "home_team": home_name,
        "away_team": away_name,
        "home_ht_goals": home_ht,
        "away_ht_goals": away_ht,
        "home_ft_goals": home_ft,
        "away_ft_goals": away_ft,
    }
    for prefix, totals in [("home", home_info["totals"]), ("away", away_info["totals"])]:
        for key, val in totals.items():
            match_row[f"{prefix}_{key}"] = val

    player_rows = []
    for is_home, team_info, team_name in [(1, home_info, home_name), (0, away_info, away_name)]:
        for p in team_info["players"]:
            player_rows.append({
                "match_id": match_id,
                "season": meta.get("season", ""),
                "team": team_name,
                "is_home": is_home,
                **p,
            })

    for prefix, is_home_val in [("home", 1), ("away", 0)]:
        if match_row.get(f"{prefix}_shots_faced_7m", 0) == 0:
            match_row[f"{prefix}_shots_faced_7m"] = sum(
                p["shots_faced_7m"] for p in player_rows if p["is_home"] == is_home_val
            )

    feuille_detaillee = len(player_rows) > 0
    return match_row, player_rows, feuille_detaillee


def _lire_csv_annexe(chemin_csv: Path, matchs_existants: list[dict], est_futur: bool) -> list[dict]:
    """Charge calendrier_futur.csv ou matchs_sans_pdf.csv en évitant tout doublon."""
    if not chemin_csv.is_file():
        return []

    saison_ref = matchs_existants[0]["season"] if matchs_existants else ""
    deja_presents = {
        (str(m.get("round", "")), _cle_equipe(m["home_team"]), _cle_equipe(m["away_team"]))
        for m in matchs_existants
    }
    ids_presents = {str(m["match_id"]) for m in matchs_existants}

    raw = chemin_csv.read_bytes()
    if not raw.strip():
        return []
    for enc in ("utf-8-sig", "cp1252"):
        try:
            texte = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        texte = raw.decode("latin-1")

    premiere_ligne = texte.splitlines()[0] if texte.splitlines() else ""
    delim = ";" if premiere_ligne.count(";") > premiere_ligne.count(",") else ","

    ajouts = []
    for row in csv.DictReader(texte.splitlines(), delimiter=delim):
        if not row.get("home_team") or not row.get("away_team"):
            continue
        cle = (str(row.get("round", "")).strip(), _cle_equipe(row["home_team"]), _cle_equipe(row["away_team"]))
        if str(row.get("match_id", "")).strip() in ids_presents or cle in deja_presents:
            continue

        if row.get("season"):
            row["season"] = re.sub(r"\s*[-/]\s*", "/", str(row["season"]).strip())
        elif saison_ref:
            row["season"] = saison_ref

        if str(row.get("round", "")).strip().isdigit():
            row["round"] = int(str(row["round"]).strip())

        if est_futur:
            row["home_ht_goals"] = ""
            row["away_ht_goals"] = ""
            row["home_ft_goals"] = ""
            row["away_ft_goals"] = ""

        ajouts.append(row)
        deja_presents.add(cle)
        if row.get("match_id"):
            ids_presents.add(str(row["match_id"]).strip())

    return ajouts


def generer_csvs(dossier_pdfs: Path, dossier_csv: Path):
    fichiers_pdf = sorted(dossier_pdfs.rglob("*.pdf")) if dossier_pdfs.exists() else []

    dossier_csv.mkdir(parents=True, exist_ok=True)
    all_matches = []
    all_players = []
    nb_scores_globaux_seuls = 0

    for pdf_path in tqdm(fichiers_pdf, desc=f"Extraction {dossier_csv.name}", unit="pdf"):
        try:
            m_row, p_rows, detaillee = parse_lnh_pdf(pdf_path)
            all_matches.append(m_row)
            all_players.extend(p_rows)
            if not detaillee:
                nb_scores_globaux_seuls += 1
                print(f"[INFO] {pdf_path.name} : feuille sans stats joueurs (score global {m_row['home_ft_goals']}-{m_row['away_ft_goals']} conservé).")
        except Exception as e:
            print(f"[ERREUR] {pdf_path.name} : {e}")

    # Fusion des matchs joués sans PDF disponible (forfaits / liens PDF LNH cassés)
    matchs_sans_pdf = _lire_csv_annexe(dossier_csv / "matchs_sans_pdf.csv", all_matches, est_futur=False)
    all_matches.extend(matchs_sans_pdf)

    # Fusion des matchs à venir
    matchs_futurs = _lire_csv_annexe(dossier_csv / "calendrier_futur.csv", all_matches, est_futur=True)
    tous_les_matchs = all_matches + matchs_futurs

    if not tous_les_matchs:
        print(f"Aucun fichier PDF trouvé dans {dossier_pdfs} et aucun CSV annexe dans {dossier_csv}")
        return

    tous_les_matchs.sort(
        key=lambda m: (
            int(m["round"]) if str(m.get("round", "")).isdigit() else 999,
            str(m.get("date", "")),
            str(m.get("time", "")),
            str(m.get("match_id", "")),
        )
    )

    match_csv_path = dossier_csv / "match.csv"
    players_csv_path = dossier_csv / "players_boxscore.csv"

    colonnes_match = list(BASE_MATCH_COLS)
    for m in tous_les_matchs:
        for k in m.keys():
            if k not in colonnes_match:
                colonnes_match.append(k)

    with match_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=colonnes_match, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(tous_les_matchs)

    if all_players:
        with players_csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_players[0].keys()))
            writer.writeheader()
            writer.writerows(all_players)

    print(
        f"Extraction terminée ({len(all_matches)} matchs joués [dont {nb_scores_globaux_seuls} sans stats joueurs, "
        f"{len(matchs_sans_pdf)} via fallback HTML/forfait], {len(matchs_futurs)} à venir, {len(all_players)} lignes joueurs) :"
    )
    print(f"  -> {match_csv_path.resolve()}")
    if all_players:
        print(f"  -> {players_csv_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Extrait les feuilles de match PDF LNH en CSV.")
    parser.add_argument("--saison", default=ANNEE, help="Saison cible (ex: 2025-2026 ou 2025/2026)")
    parser.add_argument("--toutes", action="store_true", help="Parser toutes les saisons présentes dans data/feuilles_de_match/")
    args = parser.parse_args()

    if args.toutes:
        dossiers = (
            sorted([
                d for d in DOSSIER_PDFS_RACINE.iterdir()
                if d.is_dir() and d.name >= "2015-2016"
            ])
            if DOSSIER_PDFS_RACINE.is_dir()
            else []
        )
        for d in dossiers:
            generer_csvs(d, DOSSIER_CSV_RACINE / d.name)
    else:
        saison_slug = args.saison.replace("/", "-")
        generer_csvs(DOSSIER_PDFS_RACINE / saison_slug, DOSSIER_CSV_RACINE / saison_slug)


if __name__ == "__main__":
    main()
