#!/usr/bin/env python3
"""Contrôle du dossier cache/ après reconstruction (étape du workflow hebdomadaire). Sortie 1 si une condition échoue :
  - cache à jour par rapport aux CSV (empreinte du contenu) ;
  - simulation de la saison en cours lancée à l'AVANT-DERNIÈRE journée jouée (K-1) ;
  - fiche journaliste présente, bâtie sur la dernière journée jouée K, avec la simulation de K-1 pour la journée précédente."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import server  # noqa: E402


def main() -> int:
    errs = []
    m, seasons = server.load_matches()
    S = seasons[-1]
    K = server.last_played_round(m, S)
    if not server.cache_fresh():
        errs.append("cache/stamp.json ne correspond pas aux CSV actuels (cache périmé).")
    try:
        st = json.loads((server.CACHE / "stamp.json").read_text(encoding="utf-8"))
        cur = st["runs"][0]
        if cur["season"] != S or int(cur["cutoff"]) != max(K - 1, 0):
            errs.append(f"Simulation principale : saison {cur['season']} journée {cur['cutoff']} ; attendu {S} journée {max(K - 1, 0)} (K={K}).")
    except Exception as e:
        errs.append(f"stamp.json illisible : {e}")
    try:
        press = json.loads((server.CACHE / "press.json").read_text(encoding="utf-8"))
        if int(press.get("last_played_round", -1)) != K:
            errs.append(f"press.json : dernière journée jouée {press.get('last_played_round')} ; attendu {K}.")
        if K >= 1 and not press.get("prev"):
            errs.append("press.json : pas de fiche « journée précédente ».")
    except Exception as e:
        errs.append(f"press.json absent ou illisible : {e}")
    if not (server.CACHE / "default.json").is_file():
        errs.append("default.json manquant.")
    if errs:
        print("[CACHE KO]\n - " + "\n - ".join(errs))
        return 1
    print(f"[CACHE OK] {S} : dernière journée jouée J{K}, simulation principale depuis J{max(K - 1, 0)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
