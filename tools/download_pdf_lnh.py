import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# CONFIG
SAISON_PAR_DEFAUT = "2026/2027"
COMPETITION_ID = "1"  # "1" = Daikin StarLigue, "2" = ProLigue, "4" = Coupe de France, "all" = Toutes
RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_SORTIE = RACINE_PROJET / "data/feuilles_de_match"
DOSSIER_CSV_RACINE = RACINE_PROJET / "data/data_csv"
DELAI_REQUETE = 0.3

# Correspondance exacte issue du sélecteur <select name="seasons_id"> de LNH.fr
SAISONS_ID = {
    "2026/2027": "40",
    "2025/2026": "39",
    "2024/2025": "37",
    "2023/2024": "36",
    "2022/2023": "34",
    "2021/2022": "33",
    "2020/2021": "32",
    "2019/2020": "31",
    "2018/2019": "30",
    "2017/2018": "29",
    "2016/2017": "28",
    "2015/2016": "27",
    "2014/2015": "26",
    "2013/2014": "25",
    "2012/2013": "24",
    "2011/2012": "3",
    "2010/2011": "1",
    "2009/2010": "2",
    "2008/2009": "8",
    "2007/2008": "7",
    "2006/2007": "6",
    "2005/2006": "5",
    "2004/2005": "4",
}

BASE_URL = "https://www.lnh.fr"
CALENDAR_URL = f"{BASE_URL}/matchs/calendrier"
AJAX_URL = f"{BASE_URL}/ajaxpost1"

MOIS_FR = {
    "janvier": "01", "février": "02", "fevrier": "02", "mars": "03", "avril": "04",
    "mai": "05", "juin": "06", "juillet": "07", "août": "08", "aout": "08",
    "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12", "decembre": "12",
}

CHAMPS_MATCH_CSV = [
    "match_id", "season", "round", "date", "time", "venue",
    "home_team", "away_team", "home_ht_goals", "away_ht_goals",
    "home_ft_goals", "away_ft_goals",
]


def creer_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
        "Referer": CALENDAR_URL,
    })
    return session


def generer_nom_fichier(match_url: str) -> str:
    """Extrait par exemple 'j01_chambery_paris.pdf' depuis '.../20262027/dsl/j01/chambery/paris'."""
    segments = [s for s in urlparse(match_url).path.split("/") if s]
    nom_base = "_".join(segments[-3:]) if len(segments) >= 3 else "_".join(segments)
    return f"{nom_base}.pdf"


def _extraire_date_heure(item) -> tuple[str, str]:
    """Extrait (AAAA-MM-JJ, HH:MM) depuis l'élément du match ou l'en-tête de date précédent."""
    texte_item = " ".join(item.stripped_strings)
    date_iso, heure = "", "20:00"

    m_heure = re.search(r"\b(\d{1,2})[h:](\d{2})\b", texte_item)
    if m_heure:
        heure = f"{int(m_heure.group(1)):02d}:{m_heure.group(2)}"

    zones = [texte_item]
    for prev in item.find_all_previous(["div", "h2", "h3", "h4", "li"], limit=6):
        if "calendars-listing-item" in prev.get("class", []):
            break
        zones.append(" ".join(prev.stripped_strings))

    for z in zones:
        m_num = re.search(r"\b(\d{2})/(\d{2})/(\d{4})\b", z)
        if m_num:
            date_iso = f"{m_num.group(3)}-{m_num.group(2)}-{m_num.group(1)}"
            break
        m_txt = re.search(r"\b(\d{1,2})\s+([a-zéû]+)\s+(20\d{2})\b", z.lower())
        if m_txt and m_txt.group(2) in MOIS_FR:
            date_iso = f"{m_txt.group(3)}-{MOIS_FR[m_txt.group(2)]}-{int(m_txt.group(1)):02d}"
            break

    return date_iso, heure


def _extraire_match_futur(item, cid: str, match_url: str, saison: str) -> dict | None:
    """Construit une ligne compatible match.csv pour un match non encore joué."""
    segments = [s for s in urlparse(match_url).path.split("/") if s]

    journee = None
    for seg in segments:
        m_j = re.match(r"^j0*(\d+)$", seg, re.IGNORECASE)
        if m_j:
            journee = int(m_j.group(1))
            break
    if journee is None:
        m_txt = re.search(r"Journée\s*0*(\d+)", " ".join(item.stripped_strings), re.IGNORECASE)
        if m_txt:
            journee = int(m_txt.group(1))
    if journee is None:
        return None

    noms = [
        el.get_text(" ", strip=True)
        for el in item.select(".team-name, .club-name, .calendars-listing-item-team .name")
        if el.get_text(strip=True)
    ]
    if len(noms) >= 2:
        home_team, away_team = noms[0], noms[1]
    elif len(segments) >= 2:
        home_team = segments[-2].replace("-", " ")
        away_team = segments[-1].replace("-", " ")
    else:
        return None

    if not home_team or not away_team or home_team == away_team:
        return None

    date_iso, heure = _extraire_date_heure(item)
    return {
        "match_id": f"LNH_{cid}",
        "season": saison,
        "round": journee,
        "date": date_iso,
        "time": heure,
        "venue": "",
        "home_team": home_team,
        "away_team": away_team,
        "home_ht_goals": "",
        "away_ht_goals": "",
        "home_ft_goals": "",
        "away_ft_goals": "",
    }


def trier_pdfs_a_telecharger(
    session: requests.Session,
    season_id: str,
    competition_id: str,
    dossier_saison: Path,
    saison: str,
) -> tuple[list[tuple[str, str]], list[dict], int, int]:
    """
    Récupère le calendrier via AJAX, filtre les matchs terminés ('finish'),
    construit l'URL directe du PDF via l'attribut id, écarte les fichiers déjà présents,
    et collecte les matchs à venir pour la simulation Monte Carlo.
    """
    resp_init = session.get(CALENDAR_URL, timeout=15)
    resp_init.raise_for_status()
    soup_init = BeautifulSoup(resp_init.text, "html.parser")
    key_input = soup_init.find("input", {"name": "key"})
    form_key = key_input["value"] if key_input else ""

    payload = {
        "seasons_id": season_id,
        "competitions_id": competition_id,
        "days_id": "all",
        "teams_id": "all",
        "univers": "matchs-6892",
        "key": form_key,
        "current_month": "all",
        "type": "all",
        "type_id": "all",
        "contents_controller": "sportsCalendars",
        "contents_action": "index_ajax",
        "cache": "yes",
        "cacheKeys": "univers,contents_controller,contents_action,type,type_id,club_period,club_scope,seasons_id,competitions_id,days_id,teams_id,current_month",
    }

    resp_ajax = session.post(
        AJAX_URL,
        data=payload,
        headers={"X-Requested-With": "XMLHttpRequest"},
        timeout=20,
    )
    resp_ajax.raise_for_status()
    soup_matches = BeautifulSoup(resp_ajax.text, "html.parser")

    a_telecharger = []
    matchs_futurs = []
    deja_presents = 0
    total_joues = 0
    cids_vus = set()

    for item in soup_matches.select(".calendars-listing-item"):
        cid = item.get("id")
        lien = item.select_one('a[href*="/calendriers/"]')
        if not cid or not lien or cid in cids_vus:
            continue

        cids_vus.add(cid)
        match_url = urljoin(BASE_URL, lien["href"])

        # Matchs non terminés (à venir ou reportés) -> exportés pour simulation_montecarlo.py
        if "finish" not in item.get("class", []):
            futur = _extraire_match_futur(item, cid, match_url, saison)
            if futur:
                matchs_futurs.append(futur)
            continue

        total_joues += 1
        pdf_url = f"{match_url}?cid={cid}&task=pdf"
        nom_pdf = generer_nom_fichier(match_url)
        chemin_pdf = dossier_saison / nom_pdf

        if chemin_pdf.exists() and chemin_pdf.stat().st_size > 0:
            deja_presents += 1
            continue

        a_telecharger.append((nom_pdf, pdf_url))

    return a_telecharger, matchs_futurs, total_joues, deja_presents


def telecharger_feuilles_saison(saison: str):
    if saison not in SAISONS_ID:
        saisons_valides = ", ".join(SAISONS_ID.keys())
        raise ValueError(f"Saison '{saison}' inconnue. Valeurs possibles : {saisons_valides}")

    season_id = SAISONS_ID[saison]
    saison_slug = saison.replace("/", "-")
    dossier_saison = DOSSIER_SORTIE / saison_slug
    dossier_saison.mkdir(parents=True, exist_ok=True)

    session = creer_session()
    print(f"Analyse du calendrier {saison} (ID={season_id})...")
    a_telecharger, matchs_futurs, total_joues, deja_presents = trier_pdfs_a_telecharger(
        session, season_id, COMPETITION_ID, dossier_saison, saison
    )

    dossier_csv = DOSSIER_CSV_RACINE / saison_slug
    dossier_csv.mkdir(parents=True, exist_ok=True)
    chemin_futur = dossier_csv / "calendrier_futur.csv"
    with chemin_futur.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CHAMPS_MATCH_CSV)
        writer.writeheader()
        writer.writerows(matchs_futurs)

    print(
        f"Tri terminé : {total_joues} matchs joués | {deja_presents} déjà présents | "
        f"{len(a_telecharger)} à télécharger | {len(matchs_futurs)} matchs à venir enregistrés "
        f"({chemin_futur}).\n"
    )

    if not a_telecharger:
        print("Dossier PDF déjà à jour, aucun téléchargement nécessaire.")
        return

    telecharges = 0
    for index, (nom_pdf, pdf_url) in tqdm(enumerate(a_telecharger, start=1)):
        chemin_pdf = dossier_saison / nom_pdf
        try:
            resp_pdf = session.get(pdf_url, timeout=20)
            resp_pdf.raise_for_status()

            if not resp_pdf.content.startswith(b"%PDF"):
                print(f"[{index}/{len(a_telecharger)}] Pas de PDF disponible pour {nom_pdf}")
                time.sleep(DELAI_REQUETE)
                continue

            chemin_pdf.write_bytes(resp_pdf.content)
            telecharges += 1

        except requests.RequestException as e:
            print(f"[{index}/{len(a_telecharger)}] Erreur réseau sur {nom_pdf} : {e}")

        time.sleep(DELAI_REQUETE)

    print("\n--- Bilan ---")
    print(f"Nouveaux PDF téléchargés : {telecharges}")
    print(f"Déjà présents (ignorés)  : {deja_presents}")
    print(f"Matchs futurs exportés   : {len(matchs_futurs)} -> {chemin_futur.resolve()}")


if __name__ == "__main__":
    telecharger_feuilles_saison(SAISON_PAR_DEFAUT)