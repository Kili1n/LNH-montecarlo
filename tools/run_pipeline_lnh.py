import csv
import time
import traceback
from pathlib import Path
import requests
from tqdm import tqdm

from download_pdf_lnh import SAISONS_ID, telecharger_feuilles_saison
from parser_pdf_lnh import generer_csvs
from scrap_players_lnh import (
    DELAI_REQUETE as DELAI_SCRAP,
    charger_joueurs_uniques,
    creer_session as creer_session_scrap,
    lire_csv_auto,
    scraper_fiche_joueur,
)

# ==============================================================================
# CONFIGURATION DE LA PIPELINE
# ==============================================================================
RACINE_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_PDFS_RACINE = RACINE_PROJET / "data/feuilles_de_match"
DOSSIER_CSV_RACINE = RACINE_PROJET / "data/data_csv"

# Listez ici les saisons à traiter en un seul lancement (ex: ["2026/2027"] ou ["2025/2026", "2026/2027"])
SAISONS_A_TRAITER = [
    "2021/2022",
    "2020/2021",
    "2019/2020",
    "2018/2019",
    "2017/2018",
    "2016/2017",
    "2015/2016",
    "2014/2015",
    "2013/2014",
    "2012/2013",
    "2011/2012",
    "2010/2011",
    "2009/2010",
    "2008/2009",
    "2007/2008",
    "2006/2007",
    "2005/2006",
    "2004/2005",
]

# Cache global des fiches joueurs pour éviter de rescraper le même joueur sur plusieurs saisons
CACHE_JOUEURS_GLOBAL = DOSSIER_CSV_RACINE / "_cache_players.csv"

CHAMPS_PLAYERS = [
    "player_id", "team", "jersey_num", "last_name", "first_name",
    "full_name", "position", "nationality", "birth_date",
    "height_cm", "weight_kg", "current_club", "url",
]


def log_temps_reel(rapport: dict, message: str, cle_rapport: str = "erreurs"):
    """Affiche un message en temps réel sans casser la barre tqdm et l'ajoute au rapport."""
    tqdm.write(f"    {message}")
    rapport[cle_rapport].append(message)


def ecrire_csv(chemin: Path, lignes: list[dict], fieldnames: list[str] | None = None):
    if not lignes and not fieldnames:
        return
    chemin.parent.mkdir(parents=True, exist_ok=True)
    colonnes = fieldnames or list(lignes[0].keys())
    with chemin.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=colonnes, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lignes)


# ==============================================================================
# ÉTAPE 1 : TÉLÉCHARGEMENT DES PDF DE LA SAISON
# ==============================================================================
def etape_1_telecharger_pdfs(saison: str, dossier_pdfs: Path, rapport: dict):
    """Délègue à download_pdf_lnh (calendrier, PDF manquants, calendrier_futur.csv, matchs_sans_pdf.csv)."""
    if saison not in SAISONS_ID:
        raise ValueError(f"Saison '{saison}' absente de SAISONS_ID.")
    avant = len(list(dossier_pdfs.glob("*.pdf"))) if dossier_pdfs.is_dir() else 0
    telecharger_feuilles_saison(saison)
    apres = len(list(dossier_pdfs.glob("*.pdf")))
    rapport["pdfs_deja_presents"] = avant
    rapport["pdfs_telecharges"] = apres - avant


# ==============================================================================
# ÉTAPE 2 : PARSING DES PDF -> match.csv & players_boxscore.csv
# ==============================================================================
def etape_2_parser_pdfs(saison: str, dossier_pdfs: Path, dossier_csv: Path, rapport: dict) -> Path:
    """Délègue à parser_pdf_lnh.generer_csvs (fusionne aussi calendrier_futur.csv et matchs_sans_pdf.csv)."""
    if not list(dossier_pdfs.glob("*.pdf")):
        raise FileNotFoundError(f"Aucun PDF trouvé dans {dossier_pdfs}")
    generer_csvs(dossier_pdfs, dossier_csv)
    boxscore_csv = dossier_csv / "players_boxscore.csv"
    if not boxscore_csv.is_file():
        raise RuntimeError(f"Aucune donnée valide extraite depuis {dossier_pdfs}")
    rapport["matchs_extraits"] = sum(1 for _ in open(dossier_csv / "match.csv", encoding="utf-8-sig")) - 1
    rapport["lignes_boxscore"] = sum(1 for _ in open(boxscore_csv, encoding="utf-8-sig")) - 1
    return boxscore_csv


# ==============================================================================
# ÉTAPE 3 : SCRAPING DES FICHES JOUEURS -> players.csv
# ==============================================================================
def charger_cache_joueurs(chemin_csv: Path) -> dict[tuple[str, str, str], dict]:
    cache = {}
    if chemin_csv.exists():
        for row in lire_csv_auto(chemin_csv):
            cle = (
                row["team"].strip().upper(),
                row["last_name"].strip().upper(),
                row["first_name"].strip().upper(),
            )
            cache[cle] = row
    return cache


def etape_3_scraper_joueurs(saison: str, boxscore_csv: Path, dossier_csv: Path, rapport: dict):
    joueurs_saison = charger_joueurs_uniques(boxscore_csv)
    players_csv = dossier_csv / "players.csv"

    # Fusion du cache global + fichier local existant pour éviter toute requête redondante
    cache = {}
    cache.update(charger_cache_joueurs(CACHE_JOUEURS_GLOBAL))
    cache.update(charger_cache_joueurs(players_csv))

    session = creer_session_scrap()
    fiches_saison = []
    nouveaux = 0

    for joueur in tqdm(joueurs_saison, desc=f"  [3/3 Joueurs {saison}]", unit="joueur"):
        cle_maj = (
            joueur["team"].strip().upper(),
            joueur["last_name"].strip().upper(),
            joueur["first_name"].strip().upper(),
        )

        if cle_maj in cache:
            fiche = dict(cache[cle_maj])
            fiche["team"] = joueur["team"]
            fiche["jersey_num"] = joueur["jersey_num"]
            fiche["last_name"] = joueur["last_name"]
            fiche["first_name"] = joueur["first_name"]
            fiches_saison.append(fiche)
            continue

        try:
            fiche = scraper_fiche_joueur(session, joueur)
            if fiche:
                cache[cle_maj] = fiche
                fiches_saison.append(fiche)
                nouveaux += 1
            else:
                log_temps_reel(
                    rapport,
                    f"[404 JOUEUR] {joueur['team']} | {joueur['last_name']} | {joueur['first_name']}",
                    cle_rapport="joueurs_introuvables",
                )
        except requests.RequestException as e:
            log_temps_reel(
                rapport,
                f"[ERREUR RÉSEAU] Scraping joueur {joueur['first_name']} {joueur['last_name']} : {e}",
            )
        except Exception as e:
            log_temps_reel(
                rapport,
                f"[ERREUR SCRAPING] Joueur {joueur['first_name']} {joueur['last_name']} : {e}",
            )

        time.sleep(DELAI_SCRAP)

    ecrire_csv(players_csv, fiches_saison, CHAMPS_PLAYERS)
    if cache:
        ecrire_csv(CACHE_JOUEURS_GLOBAL, list(cache.values()), CHAMPS_PLAYERS)

    rapport["joueurs_uniques"] = len(joueurs_saison)
    rapport["joueurs_scrapes_total"] = len(fiches_saison)
    rapport["joueurs_nouveaux"] = nouveaux


# ==============================================================================
# ORCHESTRATION PAR SAISON
# ==============================================================================
def executer_pipeline(saisons: list[str]):
    debut_global = time.time()
    rapports = []

    for saison in saisons:
        slug_saison = saison.replace("/", "-")
        dossier_pdfs = DOSSIER_PDFS_RACINE / slug_saison
        dossier_csv = DOSSIER_CSV_RACINE / slug_saison

        rapport = {
            "saison": saison,
            "statut": "SUCCÈS",
            "matchs_calendrier": 0,
            "pdfs_deja_presents": 0,
            "pdfs_telecharges": 0,
            "matchs_extraits": 0,
            "lignes_boxscore": 0,
            "joueurs_uniques": 0,
            "joueurs_scrapes_total": 0,
            "joueurs_nouveaux": 0,
            "joueurs_introuvables": [],
            "erreurs": [],
        }

        print(f"\n========================================================")
        print(f"SAISON {saison} -> {dossier_csv}")
        print(f"========================================================")

        try:
            etape_1_telecharger_pdfs(saison, dossier_pdfs, rapport)
            boxscore_csv = etape_2_parser_pdfs(saison, dossier_pdfs, dossier_csv, rapport)
            etape_3_scraper_joueurs(saison, boxscore_csv, dossier_csv, rapport)

            if rapport["erreurs"] or rapport["joueurs_introuvables"]:
                rapport["statut"] = "SUCCÈS PARTIEL"

        except Exception as e:
            rapport["statut"] = "ÉCHEC CRITIQUE"
            msg_critique = f"[ÉCHEC CRITIQUE] Arrêt saison {saison} : {e}"
            print(f"    {msg_critique}")
            rapport["erreurs"].append(msg_critique)
            traceback.print_exc()

        rapports.append(rapport)

    # Bilan final consolidé
    duree = round(time.time() - debut_global, 1)
    print("\n========================================================")
    print(f"BILAN FINAL DE LA PIPELINE ({duree}s)")
    print("========================================================")
    for r in rapports:
        print(
            f"[{r['statut']}] {r['saison']} : "
            f"{r['matchs_extraits']} matchs | "
            f"{r['lignes_boxscore']} boxscores | "
            f"{r['joueurs_scrapes_total']}/{r['joueurs_uniques']} joueurs "
            f"(+{r['joueurs_nouveaux']} nouveaux)"
        )
        for msg in r["joueurs_introuvables"]:
            print(f"    {msg}")
        for err in r["erreurs"]:
            print(f"    {err}")


if __name__ == "__main__":
    executer_pipeline(SAISONS_A_TRAITER)