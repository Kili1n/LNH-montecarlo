import csv
import re
import time
import unicodedata
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# CONFIG
RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_CSV = RACINE_PROJET / "data/data_csv"
INPUT_CSV = DOSSIER_CSV / "players_boxscore.csv"
OUTPUT_CSV = DOSSIER_CSV / "players.csv"

BASE_PLAYER_URL = "https://www.lnh.fr/lnh/joueurs"
DELAI_REQUETE = 0.3

# Exceptions manuelles : ("TEAM", "LAST_NAME", "FIRST_NAME") en majuscules -> "url_exacte"
URLS_MANUELLES = {
    ("PARIS", "ELISSON", "BJARKI MAR"): "https://www.lnh.fr/daikin-starligue/joueurs/bjarki-mar-elisson-3253",
    ("NANTES", "NYATEU", "O'BRIAN"): "https://www.lnh.fr/joueurs/o-brian-nyateu",
    ("CHARTRES", "ŠIŠKO", "PETER"): "https://www.lnh.fr/lnh/joueurs/peter-i-ko",
}


def creer_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Referer": "https://www.lnh.fr/",
    })
    return session


def strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def format_slug(first_name: str, last_name: str, sans_accents: bool = False) -> str:
    fn = first_name.strip()
    ln = last_name.strip()
    if sans_accents:
        fn = strip_accents(fn)
        ln = strip_accents(ln)
    slug = f"{fn}-{ln}"
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")


def est_nom_valide_pour_slug(first_name: str, last_name: str) -> bool:
    """Renvoie False si le nom/prénom contient un caractère non-ASCII ou une apostrophe."""
    texte = f"{first_name} {last_name}"
    if "'" in texte or "’" in texte:
        return False
    return texte.isascii()


def generer_urls_candidates(joueur: dict) -> list[str]:
    """Retourne l'URL manuelle si définie, ou les variantes si le nom est ASCII sans apostrophe."""
    cle_maj = (
        joueur["team"].strip().upper(),
        joueur["last_name"].strip().upper(),
        joueur["first_name"].strip().upper(),
    )
    if cle_maj in URLS_MANUELLES:
        return [URLS_MANUELLES[cle_maj]]

    if not est_nom_valide_pour_slug(joueur["first_name"], joueur["last_name"]):
        return []

    urls = []
    ln_complet = joueur["last_name"].strip()
    ln_sans_article = re.sub(r"(?:^|\b)(?:l'|le|la|les)\b\s*", " ", ln_complet, flags=re.IGNORECASE).strip()
    ln_sans_particule = re.sub(r"(?:^|\b)(?:l'|de|du|des|le|la|les)\b\s*", " ", ln_complet, flags=re.IGNORECASE).strip()

    for ln_variant in [ln_complet, ln_sans_article, ln_sans_particule]:
        if not ln_variant:
            continue
        slug = format_slug(joueur["first_name"], ln_variant, sans_accents=True)
        url = f"{BASE_PLAYER_URL}/{slug}"
        if url not in urls:
            urls.append(url)

    return urls

def date_fr_to_iso(val: str) -> str:
    m = re.match(r"^(\d{2})/(\d{2})/(\d{4})$", val.strip())
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else val.strip()


def extract_int_or_empty(val: str):
    m = re.search(r"\d+", val)
    return int(m.group(0)) if m else ""


def lire_csv_auto(csv_path: Path) -> list[dict]:
    """Lit un CSV en détectant automatiquement l'encodage (UTF-8 / cp1252) et le séparateur (, / ;)."""
    raw = csv_path.read_bytes()
    for enc in ("utf-8-sig", "cp1252"):
        try:
            texte = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        texte = raw.decode("utf-8", errors="replace")

    lignes = texte.splitlines()
    delimiteur = ";" if lignes and lignes[0].count(";") > lignes[0].count(",") else ","
    return list(csv.DictReader(lignes, delimiter=delimiteur))


def charger_joueurs_uniques(csv_path: Path) -> list[dict]:
    vus = set()
    joueurs = []
    for row in lire_csv_auto(csv_path):
        cle = (row["team"].strip(), row["last_name"].strip(), row["first_name"].strip())
        if cle in vus:
            continue
        vus.add(cle)
        joueurs.append({
            "team": row["team"].strip(),
            "jersey_num": int(row["jersey_num"]),
            "last_name": row["last_name"].strip(),
            "first_name": row["first_name"].strip(),
        })
    return joueurs


def scraper_fiche_joueur(session: requests.Session, joueur: dict) -> dict | None:
    col_infos = None
    url_valide = ""

    for url in generer_urls_candidates(joueur):
        resp = session.get(url, timeout=15)
        if resp.status_code != 200:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        bloc = soup.select_one("div.col-infos")
        if bloc and bloc.select_one(".player-name-row h2"):
            col_infos = bloc
            url_valide = resp.url
            break

    if col_infos is None:
        return None

    h2 = col_infos.select_one(".player-name-row h2")
    full_name = h2.get_text(strip=True) if h2 else ""

    player_id = ""
    audio_btn = col_infos.select_one(".player-audio-trigger[data-player-audio]")
    if audio_btn:
        m_id = re.search(r"/(\d+)-[^/]+\.mp3", audio_btn["data-player-audio"])
        if m_id:
            player_id = int(m_id.group(1))
    if player_id == "":
        m_url_id = re.search(r"-(\d+)/?$", url_valide)
        if m_url_id:
            player_id = int(m_url_id.group(1))

    pos_div = col_infos.select_one(".position")
    position = pos_div.get_text(strip=True) if pos_div else ""

    infos = {
        "nationality": "",
        "birth_date": "",
        "height_cm": "",
        "weight_kg": "",
        "current_club": "",
    }

    for row in col_infos.select(".listing-infos .row-infos"):
        lbl_el = row.select_one(".col-label")
        val_el = row.select_one(".col-value")
        if not lbl_el or not val_el:
            continue

        label = lbl_el.get_text(strip=True).rstrip(":").strip().lower()
        val = val_el.get_text(strip=True)

        if label == "nationalité":
            infos["nationality"] = val
        elif label == "né le":
            infos["birth_date"] = date_fr_to_iso(val)
        elif label == "taille":
            infos["height_cm"] = extract_int_or_empty(val)
        elif label == "poids":
            infos["weight_kg"] = extract_int_or_empty(val)
        elif label == "club actuel":
            infos["current_club"] = val

    return {
        "player_id": player_id,
        "team": joueur["team"],
        "jersey_num": joueur["jersey_num"],
        "last_name": joueur["last_name"],
        "first_name": joueur["first_name"],
        "full_name": full_name,
        "position": position,
        **infos,
        "url": url_valide,
    }


def sauvegarder_players_csv(resultats: list[dict]):
    fieldnames = [
        "player_id", "team", "jersey_num", "last_name", "first_name",
        "full_name", "position", "nationality", "birth_date",
        "height_cm", "weight_kg", "current_club", "url",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resultats)


def main():
    if not INPUT_CSV.exists():
        print(f"Fichier introuvable : {INPUT_CSV}")
        return

    joueurs = charger_joueurs_uniques(INPUT_CSV)

    deja_scrapes = {}
    if OUTPUT_CSV.exists():
        for row in lire_csv_auto(OUTPUT_CSV):
            cle = (row["team"], row["last_name"], row["first_name"])
            deja_scrapes[cle] = row

    session = creer_session()
    print(f"{len(joueurs)} joueurs uniques détectés | {len(deja_scrapes)} déjà dans {OUTPUT_CSV.name}.\n")
    introuvables = []
    nouveaux = 0

    for joueur in tqdm(joueurs, desc="Scraping joueurs", unit="joueur"):
        cle = (joueur["team"], joueur["last_name"], joueur["first_name"])
        if cle in deja_scrapes:
            continue

        try:
            fiche = scraper_fiche_joueur(session, joueur)
            if fiche:
                deja_scrapes[cle] = fiche
                nouveaux += 1
            else:
                introuvables.append(f"{joueur['team']} | {joueur['last_name']} | {joueur['first_name']}")
        except requests.RequestException as e:
            tqdm.write(f"[ERREUR RÉSEAU] {joueur['first_name']} {joueur['last_name']} : {e}")

        time.sleep(DELAI_REQUETE)

    if deja_scrapes:
        sauvegarder_players_csv(list(deja_scrapes.values()))

    print("\n--- Bilan ---")
    print(f"Nouvelles fiches scrapées : {nouveaux}")
    print(f"Total dans players.csv    : {len(deja_scrapes)}")
    if introuvables:
        print(f"Introuvables ({len(introuvables)}) :")
        for nom in introuvables:
            print(f"  - {nom}")


if __name__ == "__main__":
    main()