#!/usr/bin/env python3
"""Mise à jour hebdomadaire des données de la saison en cours (utilisée par .github/workflows/weekly-update.yml).

Enchaîne les trois outils existants, sans interaction :
  1. download_pdf_lnh.py   : calendrier LNH -> nouvelles feuilles de match PDF + calendrier_futur.csv + matchs_sans_pdf.csv
  2. parser_pdf_lnh.py     : PDF -> match.csv et players_boxscore.csv
  3. scrap_players_lnh.py  : fiches des NOUVEAUX joueurs -> players.csv (les autres viennent du fichier existant)

MODE ÉCHEC : au moindre doute (calendrier vide ou en recul, match joué manquant, CSV qui régresse, erreur réseau/parsing), le script
RESTAURE les CSV d'origine, écrit le motif dans update_report.md et sort avec le code 1. Le workflow ne commite alors rien : le site
continue de servir les données et le cache précédents.

Codes de sortie : 0 = succès (avec ou sans nouvelles données ; le workflow décide de reconstruire d'après `git status`) ; 1 = échec.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import io
import json
import sys
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import download_pdf_lnh as dl            # noqa: E402
import parser_pdf_lnh as pp              # noqa: E402

DATA_CSV = ROOT / "data" / "data_csv"
DATA_PDF = ROOT / "data" / "feuilles_de_match"
REPORT = ROOT / "update_report.md"


class UpdateError(RuntimeError):
    """Échec franc : le workflow s'arrête et signale le problème."""


def current_season(today: dt.date | None = None) -> str:
    """Saison en cours selon la date : la saison de handball commence en septembre (AAAA/AAAA+1)."""
    d = today or dt.date.today()
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start}/{start + 1}"


def played_stats(csv_path: Path) -> tuple[int, int]:
    """(matchs joués, dernière journée jouée) d'un match.csv."""
    if not csv_path.is_file():
        return 0, 0
    d = pd.read_csv(csv_path, encoding="utf-8-sig")
    ok = d["home_ft_goals"].notna() & d["away_ft_goals"].notna()
    return int(ok.sum()), (int(d.loc[ok, "round"].max()) if ok.any() else 0)


def log(msg: str) -> None:
    print(msg, flush=True)


def run_update(saison: str, skip_download: bool, skip_players: bool, dry_run: bool) -> dict:
    slug = saison.replace("/", "-")
    csv_dir, pdf_dir = DATA_CSV / slug, DATA_PDF / slug
    if saison not in dl.SAISONS_ID:
        raise UpdateError(f"Saison {saison} absente de SAISONS_ID (tools/download_pdf_lnh.py) : ajouter l'identifiant LNH de la nouvelle saison.")
    csv_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    before_played, before_round = played_stats(csv_dir / "match.csv")
    before_pdfs = len(list(pdf_dir.glob("*.pdf")))
    rep = {"saison": saison, "avant": {"joues": before_played, "journee": before_round, "pdfs": before_pdfs}}
    log(f"== {saison} : {before_played} matchs joués (J{before_round}), {before_pdfs} PDF présents")

    backups = {}
    for name in ("match.csv", "players_boxscore.csv", "players.csv", "calendrier_futur.csv", "matchs_sans_pdf.csv"):
        f = csv_dir / name
        if f.is_file():
            backups[f] = f.read_bytes()

    def restore() -> None:
        for f, content in backups.items():
            f.write_bytes(content)

    try:
        # ---- 1. calendrier + feuilles de match --------------------------------------------------------------------
        session = dl.creer_session()
        if not skip_download:
            try:
                a_tel, futurs, total_joues, deja = dl.trier_pdfs_a_telecharger(session, dl.SAISONS_ID[saison], dl.COMPETITION_ID, pdf_dir, saison)
            except Exception as e:
                raise UpdateError(f"Calendrier LNH inaccessible ou illisible (site modifié ? réseau ?) : {e}") from e
            log(f"   calendrier LNH : {total_joues} matchs terminés, {len(futurs)} à venir, {len(a_tel)} PDF à télécharger")
            if total_joues == 0 and before_played > 0:
                raise UpdateError(f"Le calendrier LNH ne renvoie aucun match terminé alors que {before_played} sont déjà connus : page modifiée ?")
            if total_joues < before_played:
                raise UpdateError(f"Le calendrier LNH n'annonce que {total_joues} matchs terminés ({before_played} déjà en base) : régression suspecte.")
            rep["calendrier"] = {"termines": total_joues, "a_venir": len(futurs), "pdfs_a_telecharger": len(a_tel)}
            if dry_run:
                log("   --dry-run : aucun téléchargement.")
                return rep
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                dl.telecharger_feuilles_saison(saison)
            log("\n".join("   " + l for l in buf.getvalue().splitlines() if l.strip())[-1800:])
            if "Erreur réseau" in buf.getvalue():
                raise UpdateError("Erreurs réseau pendant le téléchargement des PDF (voir le journal) : on réessaiera au prochain passage.")
        else:
            total_joues = None

        n_pdfs = len(list(pdf_dir.glob("*.pdf")))
        rep["pdfs_apres"] = n_pdfs

        # ---- 2. parsing -> CSV --------------------------------------------------------------------------------------
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pp.generer_csvs(pdf_dir, csv_dir)
        out = buf.getvalue()
        errs = [l for l in out.splitlines() if l.startswith("[ERREUR]")]
        if errs:
            raise UpdateError("Erreurs de parsing PDF :\n" + "\n".join(errs[:10]))
        kept = preserve_manual_fixes(backups.get(csv_dir / "players_boxscore.csv"), csv_dir / "players_boxscore.csv")
        if kept:
            log(f"   {kept} numéro(s) de maillot corrigé(s) à la main conservé(s)")
        after_played, after_round = played_stats(csv_dir / "match.csv")
        rep["apres"] = {"joues": after_played, "journee": after_round}
        log(f"   CSV : {after_played} matchs joués (J{after_round})")

        # ---- 3. contrôles de cohérence ------------------------------------------------------------------------------
        if after_played < before_played:
            raise UpdateError(f"Régression : {after_played} matchs joués après mise à jour contre {before_played} avant.")
        if total_joues is not None and after_played < total_joues:
            raise UpdateError(f"{total_joues - after_played} match(s) terminé(s) au calendrier LNH mais absents du CSV (PDF illisible ou manquant).")
        m = pd.read_csv(csv_dir / "match.csv", encoding="utf-8-sig")
        dup = m[m.duplicated(["round", "home_team", "away_team"], keep=False)]
        if len(dup):
            raise UpdateError(f"Matchs en double dans match.csv :\n{dup[['round', 'home_team', 'away_team']].head(5).to_string(index=False)}")
        pl = m[m["home_ft_goals"].notna()]
        if len(pl) and ((pl["home_ft_goals"] < 0) | (pl["away_ft_goals"] < 0) | (pl["home_ft_goals"] > 80) | (pl["away_ft_goals"] > 80)).any():
            raise UpdateError("Scores aberrants dans match.csv (négatifs ou > 80).")
        b = pd.read_csv(csv_dir / "players_boxscore.csv", encoding="utf-8-sig")
        n_box = b["match_id"].nunique()
        if n_box < 0.9 * after_played:
            raise UpdateError(f"Feuilles joueurs présentes pour {n_box} matchs seulement sur {after_played} joués : parsing incomplet.")
        rep["boxscore_matchs"] = int(n_box)

        # ---- 4. fiches des nouveaux joueurs (non bloquant) ---------------------------------------------------------------
        if not skip_players:
            try:
                import run_pipeline_lnh as rp
                r = {"erreurs": [], "joueurs_introuvables": []}
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rp.etape_3_scraper_joueurs(saison, csv_dir / "players_boxscore.csv", csv_dir, r)
                rep["joueurs"] = {k: r.get(k) for k in ("joueurs_uniques", "joueurs_scrapes_total", "joueurs_nouveaux")}
                rep["joueurs_introuvables"] = len(r["joueurs_introuvables"])
                log(f"   joueurs : {r.get('joueurs_nouveaux', 0)} nouveau(x), {len(r['joueurs_introuvables'])} introuvable(s)")
            except Exception as e:      # players.csv n'apporte que le poste et le club actuel : on continue
                rep["joueurs_erreur"] = str(e)
                log(f"   [AVERTISSEMENT] fiches joueurs non mises à jour : {e}")

        rep["nouveaux_matchs"] = after_played - before_played
        rep["nouveaux_pdfs"] = n_pdfs - before_pdfs
        return rep
    except Exception:
        restore()
        raise


def preserve_manual_fixes(old: bytes | None, new_csv: Path) -> int:
    """Le parseur se trompe parfois sur les numéros de maillot à 2 chiffres (« 72 » lu « 35 ») ; ces numéros ont été corrigés à la main dans
    players_boxscore.csv. Le re-parsing hebdomadaire de tous les PDF ne doit pas défaire ces corrections : pour un joueur déjà présent dans
    un match, on conserve l'ancien numéro (toutes les autres colonnes viennent du nouveau parsing). Renvoie le nombre de numéros conservés."""
    import csv
    if not old or not new_csv.is_file():
        return 0
    key = lambda r: (r["match_id"], r["team"], r["last_name"], r["first_name"])
    old_rows = {key(r): r["jersey_num"] for r in csv.DictReader(old.decode("utf-8-sig").splitlines())}
    with new_csv.open(newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        fields, rows = rd.fieldnames, list(rd)
    kept = 0
    for r in rows:
        j = old_rows.get(key(r))
        if j is not None and j != r["jersey_num"]:
            r["jersey_num"] = j
            kept += 1
    if kept:
        with new_csv.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return kept


def write_report(rep: dict, ok: bool, error: str = "") -> None:
    lines = [f"# Mise à jour hebdomadaire — {'succès' if ok else 'ÉCHEC'}", ""]
    if error:
        lines += ["```", error, "```", ""]
    lines.append("```json")
    lines.append(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    lines.append("```")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Mise à jour hebdomadaire des CSV de la saison en cours")
    ap.add_argument("--saison", default=None, help="AAAA/AAAA (défaut : saison en cours d'après la date)")
    ap.add_argument("--skip-download", action="store_true", help="n'appelle pas le site LNH : reparse les PDF déjà présents (tests locaux)")
    ap.add_argument("--skip-players", action="store_true", help="ne scrape pas les fiches des nouveaux joueurs")
    ap.add_argument("--dry-run", action="store_true", help="lit le calendrier LNH et s'arrête avant tout téléchargement")
    args = ap.parse_args()
    saison = (args.saison or current_season()).replace("-", "/")
    rep: dict = {"saison": saison}
    try:
        rep = run_update(saison, args.skip_download, args.skip_players, args.dry_run)
    except UpdateError as e:
        write_report(rep, False, str(e))
        print(f"\n[ÉCHEC] {e}", file=sys.stderr)
        return 1
    except Exception:
        err = traceback.format_exc()
        write_report(rep, False, err)
        print("\n[ÉCHEC] erreur inattendue :\n" + err, file=sys.stderr)
        return 1
    write_report(rep, True)
    if rep.get("nouveaux_matchs", 0) == 0 and rep.get("nouveaux_pdfs", 0) == 0 and not args.dry_run:
        log("Aucun nouveau match depuis la dernière mise à jour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
