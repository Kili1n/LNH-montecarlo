# La Starligue par les nombres — Simulation Monte Carlo & Datastorytelling

Système complet d'analyse statistique, de prévision probabiliste et de narration visuelle dédié au championnat de France masculin de handball (**LNH Starligue**).

Le projet associe un **moteur de simulation Monte Carlo multi-saisons en Python** (modèle de Dixon & Coles adapté au handball) à une **application web interactive autonome** inspirée du journalisme de données (*The Pudding*) permettant d'explorer à la fois l'historique du championnat et des milliers de futurs simulés.

---

## Architecture du projet

```text
.
├── simulation_montecarlo.py   # Moteur statistique & simulateur Monte Carlo (Python)
├── handball_storytelling.html # Application web interactive de datastorytelling (HTML/JS/SVG)
├── data/                      # (Optionnel) Archives historiques par saison ou à la racine
│   ├── 2021-2022/
│   │   ├── match.csv
│   │   ├── players.csv
│   │   └── players_boxscore.csv
│   └── ...
└── output/                    # Fichiers générés automatiquement par le simulateur
    ├── summary_storytelling.json
    ├── team_summary.parquet      (ou .csv.gz)
    ├── sim_standings.parquet     (ou .csv.gz)
    └── sim_matches.parquet       (ou .csv.gz)
```

---

## Fonctionnalités principales

### 1. Moteur de simulation (`simulation_montecarlo.py`)
* **Modèle Dixon & Coles adapté au handball** : estimation conjointe de l'efficacité offensive (`offRTG`), de l'efficacité défensive (`defRTG`) et du rythme de jeu (`Pace`, en possessions par match) par maximum de vraisemblance régularisé (Newton-Raphson).
* **Apprentissage multi-saisons & inter-saisons** : pondération exponentielle des matchs passés, retour à la moyenne inter-saisons modulé par la rétention d'effectif (mercato via `current_club`), et calibration empirique spécifique pour les clubs promus.
* **Prise en compte du contexte sportif** :
  * Impact des absences de joueurs majeurs distinguant l'attaque et la défense/les gardiens, avec décroissance temporelle.
  * Forme récente exponentielle, avantage terrain propre à chaque salle et différentiel de jours de repos.
  * Mise à jour dynamique des forces match après match au cours de la saison simulée.
* **Règlement officiel LNH** : calcul des classements avec barème à 2 points et départage exact des égalités (mini-championnat en confrontations directes avec réapplication récursive, différence de buts particulière puis générale).
* **Export haute performance** : écriture en streaming par lots au format **Parquet (ZSTD)** ou **CSV compressé (`.csv.gz`)**, accompagnée d'une synthèse **JSON**.

### 2. Interface de datastorytelling (`handball_storytelling.html`)
Application monopage **100 % locale** (aucun serveur backend requis) dotée d'un lecteur en streaming capable de décompresser et d'analyser plusieurs millions de lignes (10 000+ simulations) directement dans le navigateur. Un jeu de données de démonstration (6 saisons + 250 simulations) est embarqué pour une exploration immédiate.

#### Onglet `01. Historique` — Archives & Tendances
* **Évolution du jeu** : courbes multi-axes des buts par match, du nombre de possessions par équipe et de l'avantage à domicile saison par saison.
* **Carte tactique (`offRTG` vs `defRTG`)** : nuage de points interactif positionnant chaque club selon son efficacité offensive et défensive, sa cadence (taille des points) et son efficacité nette (`netRTG`).
* **Trajectoires & dynasties (Bump Chart)** : évolution des classements finaux sur plusieurs saisons avec isolation interactive d'un club.
* **Individualités & feuilles de match** : classements et fiches des meilleurs buteurs (réussite au tir, jets de 7m), gardiens (% d'arrêts) et meilleurs joueurs à l'évaluation LNH.
* **Chapitres analytiques complémentaires** : palmarès historique, livre des records, plus longues séries (victoires, invincibilité, défaites), performances domicile vs extérieur, matrice des scores exacts, **classement Elo historique** (pics de forme et plus gros exploits) et matrice des rivalités directes.

#### Onglet `02. Simulation Monte Carlo` — Exploration Probabiliste
* **Distribution des destins** : tableau complet des projections avec intervalles de confiance à 90 % (`Pts_P05` – `Pts_P95`) et probabilités d'objectifs (Titre, Podium, Top 4, Maintien, Relégation).
* **Détection du « Cygne Noir »** : identification automatique de la simulation la plus aberrante statistiquement (écart quadratique normalisé sur les points/rangs et surprise des résultats), comparée côte à côte à la simulation la plus typique, avec extraction des **6 séismes** (matchs simulés les plus improbables).
* **Sensibilité des effectifs** : croisement entre volatilité des classements Monte Carlo et dépendance au duo majeur de chaque club.
* **Zoom par club & Panorama** : histogrammes des points et des rangs finaux, matrice thermique des classements, seuils de points requis pour chaque objectif, matrice des duels ($P(\text{club A devant club B})$) et calibration des probabilités de match.
* **Rejouer une saison simulée** : outil permettant de tirer une trajectoire (au hasard, pire début de saison, meilleur début, plus longue série de défaites) pour un club et un rang final donnés, et de revivre sa saison journée par journée.

---

## Méthodologie statistique

1. **Reconstitution des possessions** : calculées à partir des feuilles de match via $\text{Possessions} = \text{Tirs} + \text{Balles perdues} - 0{,}5 \times \text{Jets de 7m}$ (avec contrôle de cohérence et imputation par efficacité moyenne si les boxscores sont incomplets).
2. **Modélisation des buts et de la dispersion** : contrairement au football (loi de Poisson), le handball présente une sous-dispersion des scores autour de l'espérance. Le simulateur utilise un paramètre de dispersion de Fano (`disp_fano = 0.58` par défaut) basculant dynamiquement entre loi Binomiale (sous-dispersion), loi de Poisson et loi Gamma-Poisson (sur-dispersion), couplé à un aléa de rythme match par match.
3. **Probabilités 1N2 exactes** : calculées par **quadrature de Gauss-Hermite à 7 points** intégrant l'incertitude sur le volume de possessions de la rencontre.
4. **Normalisation canonique des clubs** : un dictionnaire intégré (`DEFAULT_TEAM_ALIASES`, extensible via `team_aliases.json` ou `team_aliases.csv`) unifie automatiquement les variantes d'appellation des clubs français (ex. *Paris Saint-Germain Handball*, *Paris SG* $\rightarrow$ `PSG`).

---

## Installation et utilisation

### Prérequis (Simulateur Python)
* Python 3.10+
* Dépendances principales :
  ```bash
  pip install numpy pandas pyarrow
  ```
  *(Note : `pyarrow` est optionnel ; en son absence, le script bascule automatiquement sur l'export `.csv.gz`.)*

### Lancer une simulation Monte Carlo

Placez vos fichiers historiques (`match.csv`, et optionnellement `players.csv` et `players_boxscore.csv`) à la racine d'un dossier ou organisés en sous-dossiers par saison (ex. `data/2024-2025/match.csv`), puis exécutez :

```bash
python simulation_montecarlo.py \
  --matches data/ \
  --boxscores data/ \
  --players data/ \
  --season 2025-2026 \
  --cutoff-round 14 \
  --n-sims 10000 \
  --output output/
```

Vous pouvez ajuster n'importe quel hyperparamètre du modèle (`ModelParams`) directement en ligne de commande via `--param` :

```bash
python simulation_montecarlo.py --matches data/ --param dc_xi=0.60 --param reapply_h2h=true
```

### Visualiser les résultats dans l'interface web

1. Ouvrez `handball_storytelling.html` dans un navigateur moderne (Chrome, Firefox, Edge, Safari).
2. Par défaut, l'interface charge un **jeu de démonstration complet**.
3. Cliquez sur **« Importer fichiers (.parquet / .csv / .gz) »** (ou glissez-déposez directement vos fichiers ou dossiers sur la page) pour charger vos propres archives historiques (`match.csv`, `players.csv`, `players_boxscore.csv`) ainsi que les fichiers générés dans `output/` (`sim_standings`, `sim_matches`, `team_summary`, `summary_storytelling.json`).

---

## Format des données

### Fichiers d'entrée attendus

| Fichier | Rôle | Colonnes clés |
| :--- | :--- | :--- |
| **`match.csv`** | Calendrier et résultats historiques / à venir *(requis)* | `match_id`, `season`, `round`, `date`, `time`, `home_team`, `away_team`, `home_ft_goals`, `away_ft_goals` *(laisser vide pour les matchs non joués)* |
| **`players_boxscore.csv`** | Statistiques individuelles par match *(optionnel)* | `match_id`, `team`, `is_home`, `first_name`, `last_name`, `seconds_played`, `goals`, `shots`, `goals_7m`, `shots_7m`, `saves`, `shots_faced`, `bp`, `br`, `tc`, `a`, `lnh_score` |
| **`players.csv`** | Référentiel des effectifs et transferts *(optionnel)* | `team`, `first_name`, `last_name`, `position`, `current_club` |

### Fichiers générés dans `output/`

| Fichier | Description |
| :--- | :--- |
| **`summary_storytelling.json`** | Métadonnées de la simulation, paramètres utilisés, ratings Dixon & Coles initiaux, métriques de backtest et identifiants des simulations notables. |
| **`team_summary.parquet` / `.csv.gz`** | Synthèse par équipe : rang moyen/médian, écart-type, points attendus, percentiles (`P05`, `P50`, `P95`), probabilités (`P_1st`, `P_Podium`, `P_Top4`, `P_Relegated`) et ratings finaux moyens. |
| **`sim_standings.parquet` / `.csv.gz`** | Classement final complet (`Rank`, `Pts`, `W`, `D`, `L`, `GF`, `GA`, `GD`) pour chacune des $S$ simulations. |
| **`sim_matches.parquet` / `.csv.gz`** | Détail de chaque match pour chaque simulation (`hg`, `ag`, `result`, probabilités prédites `p_home`/`p_draw`/`p_away`, ratings dynamiques et indicateur `frozen`). |

---

## Notes techniques

* **Gestion mémoire côté navigateur** : `handball_storytelling.html` utilise `DecompressionStream('gzip')` pour lire les archives `.csv.gz` ligne par ligne, `hyparquet` pour la lecture par *row-groups* des fichiers `.parquet`, et des tableaux typés (`Int16Array`, `Float32Array`, `Uint8Array`) afin de stocker des millions de matchs simulés avec une empreinte mémoire minimale (~25 octets par match).
* **Export Parquet en Python** : dans `simulation_montecarlo.py`, si `pyarrow` est installé dans votre environnement et que vous souhaitez forcer l'export en `.parquet` plutôt qu'en `.csv.gz`, vérifiez que le drapeau `HAS_PARQUET` en tête de script est bien défini à `True`.