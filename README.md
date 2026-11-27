# La Starligue en chiffres — Simulation Monte Carlo du championnat de handball

**Démo en ligne : https://lnh.fokalpress.fr**

Projet personnel de data science / ingénierie logicielle : à partir de 20 saisons de la **LNH Starligue** (championnat de France masculin de handball), j'ai construit **de bout en bout** une chaîne qui collecte les données, estime un modèle statistique, simule des dizaines de milliers de fins de saison, puis restitue les résultats dans une application web interactive.

> **La question posée :** *« Quelle est la probabilité que tel club soit champion, finisse sur le podium ou descende ? »* — et, au-delà : *quel est l'enjeu réel du prochain match pour chaque équipe ?*

---

## Ce que le projet démontre

| Compétence | Mise en œuvre |
| :--- | :--- |
| **Collecte & nettoyage de données** | Téléchargement des feuilles de match officielles (PDF), extraction par parsing de la mise en page, scraping des fiches joueurs, normalisation des noms de clubs → ~60 fichiers CSV sur 20 saisons |
| **Modélisation statistique** | Modèle de type Dixon & Coles adapté au handball, estimé par maximum de vraisemblance régularisé (Newton-Raphson) |
| **Simulation Monte Carlo** | 10 000 saisons simulées match par match, avec application exacte du règlement de départage de la ligue |
| **Calcul performant** | NumPy vectorisé, export en streaming, garde-fous mémoire pour tourner sur un hébergement à 512 Mo |
| **Backend & déploiement** | Serveur HTTP Python (API JSON, cache pré-calculé, ETag + gzip), déployé sur Render avec nom de domaine personnalisé |
| **Visualisation / data storytelling** | Application web monopage (JS + SVG, sans framework) : graphiques interactifs |

---

## Le site

L'application s'ouvre sur l'onglet **Simulation Monte Carlo**, avec une simulation déjà calculée.

### 1. Simulation Monte Carlo
* **Probabilités par club** : titre, podium, top 4, maintien, relégation, avec intervalles de confiance à 90 % sur les points finaux.
* **Cygne noir** : détection automatique de la simulation la plus improbable et des 6 « séismes » (matchs simulés les plus surprenants).
* **Panorama** : histogrammes des rangs et des points, matrice thermique des classements, seuils de points nécessaires pour chaque objectif, matrice de duels (*P(club A termine devant club B)*), calibration des probabilités de match.
* **Scénarios conditionnels** : on « verrouille » des résultats de matchs à venir et on voit comment les probabilités se déplacent (conditionnement sur les simulations compatibles).
* **Rejouer une saison simulée** : revivre, journée par journée, une trajectoire extrême (pire début, plus longue série de défaites…).

### 2. Historique
Évolution du jeu saison par saison, carte tactique attaque/défense, trajectoires de classement (bump chart), meilleurs buteurs et gardiens, palmarès, records, séries, classement Elo historique, rivalités.

### 3. Fiche journaliste
Un document imprimable (PDF) avant chaque journée : l'affiche, l'enjeu chiffré au classement, les caps et records en vue, le face-à-face. Les probabilités viennent des 10 000 simulations lancées à la bonne journée.

---

## Méthodologie

1. **Efficacité offensive / défensive.** Pour chaque club, on estime une force d'attaque (`offRTG`), de défense (`defRTG`) et un rythme de jeu, exprimés en buts par match, par maximum de vraisemblance régularisé. Les matchs récents pèsent plus que les anciens (pondération exponentielle).
2. **Passage d'une saison à l'autre.** Les forces sont ramenées vers la moyenne, plus ou moins fortement selon la **rétention de l'effectif** (mercato) ; une calibration spécifique traite les clubs promus.
3. **Contexte du match.** Avantage du terrain propre à chaque salle, forme récente, jours de repos, absences de joueurs majeurs (attaque et défense/gardiens distinguées).
4. **Pourquoi pas une loi de Poisson ?** Au handball, les scores sont *sous-dispersés* par rapport à Poisson (beaucoup de possessions, peu de hasard par possession). Le simulateur utilise un paramètre de dispersion (Fano) qui bascule entre loi Binomiale, Poisson et Gamma-Poisson.
5. **Probabilités 1N2** calculées par **quadrature de Gauss-Hermite** (7 points) pour intégrer l'incertitude sur le rythme du match.
6. **Mise à jour dynamique.** Pendant la saison simulée, les forces sont réajustées match après match.
7. **Règlement officiel.** Barème à 2 points ; égalités départagées par mini-championnat en confrontations directes (réappliqué récursivement), puis différence de buts particulière, puis générale.
8. **Validation.** Un mode de *backtest* (`--backtest-seasons`) rejoue des saisons passées pour comparer prévisions et résultats réels.
9. **Absences aléatoires (optionnel, désactivé par défaut).** `--param inj_enabled=true` tire dans chaque simulation, indépendamment, des absences de joueurs (blessures et indisponibilités) : taux de 10,8 % par journée pour un joueur régulier et durées issues d'une estimation de Kaplan-Meier sur 7 532 absences observées entre 2007 et 2026. L'effet passe par le poids offensif et défensif du joueur dans son équipe, centré sur l'absentéisme moyen déjà contenu dans les ratings. Mesuré sur un backtest de 19 saisons, il n’améliore pas la précision (légère dégradation du score de probabilité du rang réel et des points attendus) : il reste donc désactivé.

---

## Architecture

```text
.
├── simulation_montecarlo.py   # Moteur statistique + simulateur (≈2 400 lignes)
├── server.py                  # Serveur HTTP : API JSON, cache, fiche journaliste
├── index.html                 # Application web (JS/SVG, sans framework)
├── cache/                     # Résultats pré-calculés (JSON), versionnés pour le déploiement
├── data/data_csv/<saison>/    # match.csv, players.csv, players_boxscore.csv
├── tools/                     # Pipeline de collecte : PDF → CSV
│   ├── download_pdf_lnh.py    #   téléchargement des feuilles de match
│   ├── parser_pdf_lnh.py      #   extraction des statistiques depuis les PDF
│   ├── scrap_players_lnh.py   #   fiches joueurs
│   └── run_pipeline_lnh.py    #   orchestration des trois étapes
└── requirements.txt
```

**Flux de données :** PDF officiels → CSV → modèle Dixon & Coles → 10 000 simulations → bundles JSON (`cache/`) → serveur → navigateur.

**Choix d'ingénierie notables**
* **Cache pré-calculé et versionné** : le site s'ouvre en une seule requête, sans recalcul. Le cache est invalidé par une empreinte du contenu des CSV (insensible aux fins de ligne et aux séparateurs de chemin Windows/Linux — un bug réel rencontré en production sur Render).
* **Contraintes mémoire** : une simulation à la fois, vérification de la mémoire libre avant lancement, plafond du nombre de simulations demandables depuis le site.
* **Chargement paresseux** : l'historique (~500 Ko) n'est chargé qu'à la première ouverture de l'onglet.

---

## Lancer le projet en local

```bash
pip install -r requirements.txt
python server.py            # puis http://localhost:8000
```

Régénérer le cache (≈ 3 min, ~250 Mo de mémoire) :

```bash
python server.py --build-cache
```

Lancer le simulateur seul :

```bash
python simulation_montecarlo.py --data-dir data/data_csv --season 2025-2026 \
  --cutoff-round 14 --n-simulations 10000 --output-dir output
```

Tout hyperparamètre du modèle se règle en ligne de commande : `--param dc_xi=0.60`.

Collecter une nouvelle saison (dépendances supplémentaires : `requests`, `beautifulsoup4`, `tqdm`, `pdfminer.six`) :

```bash
python tools/run_pipeline_lnh.py
```

### Déploiement
Hébergé sur **Render** (service web Python, démarrage `python server.py`), code sur GitHub, domaine personnalisé. Variables utiles : `CACHE_SIMS`, `MAX_SIMS`, `MIN_FREE_MB`, `AUTO_BUILD_CACHE`.

---

## Format des données

| Fichier | Rôle | Colonnes clés |
| :--- | :--- | :--- |
| `match.csv` | Calendrier et résultats | `match_id`, `season`, `round`, `date`, `home_team`, `away_team`, `home_ft_goals`, `away_ft_goals` (vide si non joué) |
| `players_boxscore.csv` | Statistiques individuelles par match | `match_id`, `team`, `goals`, `shots`, `goals_7m`, `saves`, `shots_faced`… |
| `players.csv` | Effectifs et transferts | `team`, `first_name`, `last_name`, `position`, `current_club` |

## Pistes d'amélioration
* Tests automatisés du moteur de départage et du modèle.
* Mise à jour automatique des données après chaque journée (tâche planifiée).
* Évaluation chiffrée de la calibration (score de Brier, diagramme de fiabilité) affichée sur le site.

*Données : feuilles de match publiques de la LNH. Projet indépendant, non affilié à la Ligue Nationale de Handball.*
