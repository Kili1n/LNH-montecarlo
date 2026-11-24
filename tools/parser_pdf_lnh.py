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
DOSSIER_PDFS = RACINE_PROJET / "data/feuilles_de_match" / ANNEE  # Ou RACINE_PROJET / "data/feuilles_de_match" pour toutes les saisons
DOSSIER_CSV = RACINE_PROJET / "data/data_csv" / ANNEE

BASE_MATCH_COLS = [
    "match_id", "season", "round", "date", "time", "venue",
    "home_team", "away_team", "home_ht_goals", "away_ht_goals",
    "home_ft_goals", "away_ft_goals",
]

# Table d'équivalence pour reconnaître qu'un match futur et un PDF concernent le même club
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


def _cle_equipe(nom: str) -> str:
    """Normalise le nom d'une équipe pour éviter tout doublon entre calendrier_futur.csv et les PDF."""
    s = unicodedata.normalize("NFKD", str(nom).strip())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().replace("’", "'")
    cle = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return ALIAS_EQUIPES.get(cle, cle)


def extract_rows(pdf_path: Path) -> list[dict]:
    """Extrait tous les caractères avec leurs coordonnées exactes et les regroupe par ligne."""
    chars = []

    def walk(obj):
        if isinstance(obj, LTChar):
            chars.append((obj.bbox[1], obj.bbox[0], obj.bbox[2], obj.get_text()))
        elif hasattr(obj, "__iter__"):
            for child in obj:
                walk(child)

    for page_layout in extract_pages(str(pdf_path)):
        walk(page_layout)
        break  # Chaque feuille de match LNH tient sur 1 page

    chars.sort(key=lambda c: (-round(c[0], 1), c[1]))
    rows = []
    for y0, x0, x1, txt in chars:
        if not rows or abs(y0 - rows[-1]["y"]) > 2.0:
            rows.append({"y": y0, "chars": [(x0, x1, txt)]})
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


def parse_lnh_pdf(pdf_path: Path) -> tuple[dict, list[dict]]:
    rows = extract_rows(pdf_path)

    meta = {}
    home_ht = away_ht = home_ft = away_ft = 0
    header_home_name = ""
    header_away_name = ""

    # 1. En-tête : métadonnées et scores (y > 700)
    for r in rows:
        y = r["y"]
        if y < 700:
            continue

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
            h_txt = cell_text(r["chars"], 50, 250)
            a_txt = cell_text(r["chars"], 350, 560)
            if h_txt and a_txt:
                header_home_name = h_txt
                header_away_name = a_txt

        center_lbl = cell_text(r["chars"], 250, 345)
        if center_lbl == "Mi-temps":
            home_ht = parse_int(cell_text(r["chars"], 80, 200))
            away_ht = parse_int(cell_text(r["chars"], 400, 520))
        elif center_lbl == "Score final":
            home_ft = parse_int(cell_text(r["chars"], 80, 200))
            away_ft = parse_int(cell_text(r["chars"], 400, 520))

    # 2. Tableaux statistiques domicile puis extérieur (de y <= 700 jusqu'à la légende)
    teams_data = []
    current_team = None
    noms_equipes_attendus = {
        n.strip().lower() for n in (header_home_name, header_away_name) if n.strip()
    }
    equipes_deja_vues = set()

    for r in rows:
        y = r["y"]
        if y > 700:
            continue
        full_line = cell_text(r["chars"], 0, 600)
        if full_line.startswith("* B.R."):
            break

        jersey_txt = cell_text(r["chars"], *COLS["jersey"])
        last_name_txt = cell_text(r["chars"], *COLS["last_name"])
        team_title_txt = cell_text(r["chars"], 21, 180)
        stats_area_txt = cell_text(r["chars"], 180, 600)

        titre_low = team_title_txt.strip().lower()

        # Détection uniquement des 2 équipes du match (ignore les officiels comme MADSEN Stefan)
        if (
            len(teams_data) < 2
            and not jersey_txt
            and not stats_area_txt
            and titre_low in noms_equipes_attendus
            and titre_low not in equipes_deja_vues
        ):
            current_team = {
                "name": team_title_txt,
                "players": [],
                "totals": {},
            }
            teams_data.append(current_team)
            equipes_deja_vues.add(titre_low)
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
            continue

        # Ligne joueur individuel
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

    if len(teams_data) < 2:
        raise ValueError(f"Structure incomplète dans {pdf_path.name}")

    home_info, away_info = teams_data[0], teams_data[1]
    home_name = home_info["name"] or header_home_name
    away_name = away_info["name"] or header_away_name
    match_id = meta.get("match_id", pdf_path.stem)

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
        if match_row[f"{prefix}_shots_faced_7m"] == 0:
            match_row[f"{prefix}_shots_faced_7m"] = sum(
                p["shots_faced_7m"] for p in player_rows if p["is_home"] == is_home_val
            )

    return match_row, player_rows


def charger_matchs_futurs(dossier_csv: Path, matchs_joues: list[dict]) -> list[dict]:
    """Charge calendrier_futur.csv et écarte les rencontres déjà extraites des PDF."""
    chemin_futur = dossier_csv / "calendrier_futur.csv"
    if not chemin_futur.is_file():
        return []

    saison_ref = matchs_joues[0]["season"] if matchs_joues else ""
    deja_joues = {
        (str(m.get("round", "")), _cle_equipe(m["home_team"]), _cle_equipe(m["away_team"]))
        for m in matchs_joues
    }
    ids_joues = {str(m["match_id"]) for m in matchs_joues}

    raw = chemin_futur.read_bytes()
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

    futurs = []
    for row in csv.DictReader(texte.splitlines(), delimiter=delim):
        if not row.get("home_team") or not row.get("away_team"):
            continue
        cle = (str(row.get("round", "")).strip(), _cle_equipe(row["home_team"]), _cle_equipe(row["away_team"]))
        if str(row.get("match_id", "")).strip() in ids_joues or cle in deja_joues:
            continue

        if saison_ref:
            row["season"] = saison_ref
        elif row.get("season"):
            row["season"] = re.sub(r"\s*[-/]\s*", "/", str(row["season"]).strip())

        if row.get("round", "").isdigit():
            row["round"] = int(row["round"])

        row["home_ht_goals"] = ""
        row["away_ht_goals"] = ""
        row["home_ft_goals"] = ""
        row["away_ft_goals"] = ""
        futurs.append(row)

    return futurs


def generer_csvs(dossier_pdfs: Path, dossier_csv: Path):
    fichiers_pdf = sorted(dossier_pdfs.rglob("*.pdf")) if dossier_pdfs.exists() else []

    dossier_csv.mkdir(parents=True, exist_ok=True)
    all_matches = []
    all_players = []

    for pdf_path in tqdm(fichiers_pdf, desc="Extraction PDF", unit="pdf"):
        try:
            m_row, p_rows = parse_lnh_pdf(pdf_path)
            all_matches.append(m_row)
            all_players.extend(p_rows)
        except Exception as e:
            print(f"[ERREUR] {pdf_path.name} : {e}")

    matchs_futurs = charger_matchs_futurs(dossier_csv, all_matches)
    tous_les_matchs = all_matches + matchs_futurs

    if not tous_les_matchs:
        print(f"Aucun fichier PDF trouvé dans {dossier_pdfs} et aucun calendrier_futur.csv dans {dossier_csv}")
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

    colonnes_match = list(all_matches[0].keys()) if all_matches else BASE_MATCH_COLS
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
        f"Extraction terminée ({len(all_matches)} matchs joués, "
        f"{len(matchs_futurs)} matchs à venir, {len(all_players)} lignes joueurs) :"
    )
    print(f"  -> {match_csv_path.resolve()}")
    if all_players:
        print(f"  -> {players_csv_path.resolve()}")


if __name__ == "__main__":
    generer_csvs(DOSSIER_PDFS, DOSSIER_CSV)