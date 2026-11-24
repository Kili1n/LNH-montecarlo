import argparse
import csv
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# CONFIG
SAISON_PAR_DEFAUT = "2021/2022"
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


def _extraire_infos_base_match(item, cid: str, match_url: str, saison: str) -> dict | None:
    """Construit une ligne compatible match.csv depuis un bloc calendrier LNH."""
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

    # Tentative de récupération d'un éventuel score affiché dans le calendrier (ex: forfait 10 - 0)
    home_ft, away_ft = "", ""
    scores_els = [
        el.get_text(strip=True)
        for el in item.select(".score, .calendars-listing-item-score, .result")
        if re.fullmatch(r"\d{1,2}", el.get_text(strip=True))
    ]
    if len(scores_els) >= 2:
        home_ft, away_ft = int(scores_els[0]), int(scores_els[1])
    else:
        m_sc = re.search(r"\b(\d{1,2})\s*[-–]\s*(\d{1,2})\b", " ".join(item.stripped_strings))
        if m_sc:
            home_ft, away_ft = int(m_sc.group(1)), int(m_sc.group(2))

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
        "home_ft_goals": home_ft,
        "away_ft_goals": away_ft,
    }


def _scraper_fallback_page_match(session: requests.Session, match_url: str, infos_base: dict) -> dict:
    """
    Quand le PDF est inaccessible (match forfait/annulé comme 20/21 ou 15/16,
    ou lien PDF buggé comme 12/13 et 13/14), récupère le score officiel sur la page HTML du match.
    """
    resultat = dict(infos_base)
    try:
        resp = session.get(match_url, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        texte = " ".join(soup.stripped_strings)

        # Code match officiel s'il apparaît sur la fiche
        m_code = re.search(r"Code\s+match\s*:?\s*([A-Z0-9_-]{4,20})", texte)
        if m_code:
            resultat["match_id"] = m_code.group(1)

        # Score final sur la fiche match
        scores = [
            int(el.get_text(strip=True))
            for el in soup.select(".match-header-score .score, .score-home, .score-away, .team-score")
            if re.fullmatch(r"\d{1,2}", el.get_text(strip=True))
        ]
        if len(scores) >= 2:
            resultat["home_ft_goals"] = scores[0]
            resultat["away_ft_goals"] = scores[1]

        # Score mi-temps éventuel "(14 - 12)"
        m_mt = re.search(r"(?:Mi-temps|MT)\s*:?\s*\(?(\d{1,2})\s*[-–]\s*(\d{1,2})\)?", texte, re.IGNORECASE)
        if m_mt:
            resultat["home_ht_goals"] = int(m_mt.group(1))
            resultat["away_ht_goals"] = int(m_mt.group(2))

        # Lieu éventuel
        m_lieu = re.search(r"Lieu\s*:?\s*([A-Za-zÀ-ÿ0-9\s'’.-]{3,50}?)(?:\s+Arbitres|\s+Affluence|\s+Journée|$)", texte)
        if m_lieu and not resultat["venue"]:
            resultat["venue"] = m_lieu.group(1).strip()
    except Exception:
        pass

    return resultat


def trier_pdfs_a_telecharger(
    session: requests.Session,
    season_id: str,
    competition_id: str,
    dossier_saison: Path,
    saison: str,
) -> tuple[list[tuple[str, str, str, dict]], list[dict], int, int]:
    """
    Récupère le calendrier via AJAX, filtre les matchs terminés ('finish'),
    construit l'URL directe du PDF via l'attribut id, écarte les fichiers déjà présents,
    et collecte les matchs à venir.
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
        infos_base = _extraire_infos_base_match(item, cid, match_url, saison)

        # Matchs non terminés (à venir ou reportés) -> exportés pour simulation_montecarlo.py
        if "finish" not in item.get("class", []):
            if infos_base:
                futur = dict(infos_base)
                futur["home_ht_goals"] = ""
                futur["away_ht_goals"] = ""
                futur["home_ft_goals"] = ""
                futur["away_ft_goals"] = ""
                matchs_futurs.append(futur)
            continue

        total_joues += 1
        pdf_url = f"{match_url}?cid={cid}&task=pdf"
        nom_pdf = generer_nom_fichier(match_url)
        chemin_pdf = dossier_saison / nom_pdf

        if chemin_pdf.exists() and chemin_pdf.stat().st_size > 0:
            deja_presents += 1
            continue

        a_telecharger.append((nom_pdf, pdf_url, match_url, infos_base or {}))

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
    matchs_sans_pdf = []

    for index, (nom_pdf, pdf_url, match_url, infos_base) in tqdm(enumerate(a_telecharger, start=1)):
        chemin_pdf = dossier_saison / nom_pdf
        try:
            resp_pdf = session.get(pdf_url, timeout=20)
            resp_pdf.raise_for_status()

            if not resp_pdf.content.startswith(b"%PDF"):
                print(f"[{index}/{len(a_telecharger)}] Pas de PDF disponible pour {nom_pdf} -> récupération score HTML (forfait/lien cassé)")
                if infos_base:
                    infos_page = _scraper_fallback_page_match(session, match_url, infos_base)
                    matchs_sans_pdf.append(infos_page)
                time.sleep(DELAI_REQUETE)
                continue

            chemin_pdf.write_bytes(resp_pdf.content)
            telecharges += 1

        except requests.RequestException as e:
            print(f"[{index}/{len(a_telecharger)}] Erreur réseau sur {nom_pdf} : {e}")
            if infos_base:
                matchs_sans_pdf.append(infos_base)

        time.sleep(DELAI_REQUETE)

    if matchs_sans_pdf:
        chemin_sans_pdf = dossier_csv / "matchs_sans_pdf.csv"
        with chemin_sans_pdf.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CHAMPS_MATCH_CSV)
            writer.writeheader()
            writer.writerows(matchs_sans_pdf)
        print(f"Matchs sans PDF (forfaits / liens LNH cassés) sauvegardés -> {chemin_sans_pdf.resolve()}")

    print("\n--- Bilan ---")
    print(f"Nouveaux PDF téléchargés     : {telecharges}")
    print(f"Déjà présents (ignorés)      : {deja_presents}")
    print(f"Sans PDF (forfait/lien cassé): {len(matchs_sans_pdf)}")
    print(f"Matchs futurs exportés       : {len(matchs_futurs)} -> {chemin_futur.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Télécharge les feuilles de match PDF LNH.")
    parser.add_argument("--saison", default=SAISON_PAR_DEFAUT, help="Saison au format AAAA/AAAA (ex: 2025/2026)")
    parser.add_argument("--toutes", action="store_true", help="Télécharger toutes les saisons configurées")
    args = parser.parse_args()

    if args.toutes:
        for s in SAISONS_ID:
            telecharger_feuilles_saison(s)
    else:
        telecharger_feuilles_saison(args.saison.replace("-", "/"))


if __name__ == "__main__":
    main()