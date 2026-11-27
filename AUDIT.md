# Audit complet — La Starligue en chiffres

Audit mené le 27/11/2026 sur les vrais CSV (20 saisons : 3 990 lignes `match.csv`, dont 3 837 matchs joués, et 117 141 lignes `players_boxscore.csv`).
Point de départ : tag git `pre-audit` (commit `66ae4c1`). Un commit par phase, listés en (b).

Outils ajoutés : `tools/audit_data.py` (audit d'intégrité des CSV, rejouable), scripts de contrôle indépendants (relecture des CSV bruts sans réutiliser le code du projet) décrits ci-dessous.

---

## (a) Ce qui a été vérifié, et comment

### Phase 0 — État des lieux
| Contrôle | Résultat |
|---|---|
| Tests existants | **Aucun** dans le dépôt (pas de dossier `tests/`). Les contrôles ci-dessous en tiennent lieu. |
| `py_compile` (server, simulateur, tools) | OK |
| `pyflakes` | 1 import inutilisé (`shutil`, corrigé) ; 2 f-strings sans variable dans `tools/run_pipeline_lnh.py` (non corrigé, sans effet) |
| `node --check` du script de `index.html` | OK |
| Démarrage du serveur sur les vrais CSV | Aucune erreur ni avertissement. Un bug de démarrage sous console Windows (caractère « → ») avait été corrigé avant l'audit. |
| Chargement à froid, mode cache | Console du navigateur **vide**, onglet Monte Carlo avec simulation déjà affichée. |

### Phase 1 — Intégrité des données (`tools/audit_data.py`)
- **Colonnes / types / manquants** : toutes les colonnes attendues sont présentes. Manquants : `date` 159 (153 matchs à venir + **6 matchs joués sans date** : 2012/13 ×2, 2013/14, 2015/16, 2020/21 ×2), scores 153 (à venir). Format de date : 100 % `AAAA-MM-JJ`.
- **Doublons** : 0 doublon `(saison, match_id)`, 0 doublon joueur×match. **180 identifiants de match partagés entre 2012/13 et 2014/15** (déjà corrigé avant l'audit par un suffixe `@2014`). 0 joueur dans deux clubs le même match.
- **Orphelins** : 0 ligne joueur sans match. **20 matchs joués sans feuille de match** (2007/08 : 13, 2012/13 : 3, 2013/14 : 1, 2015/16 : 1, 2020/21 : 2) — gérés à l'affichage (« Pas de feuille de match détaillée »).
- **Clubs** : 41 graphies brutes → 28 clubs. **Une variante non unifiée trouvée** : `ILLEFRANCHE SUR SAÔNE` / `Villefranche sur Saône` en 2007/08 = deux clubs fantômes (15 clubs au lieu de 14). Corrigé.
- **Homonymes / graphies de joueurs** : 0 nom avec deux dates de naissance dans `players.csv` ; 0 variante de graphie après normalisation ; 45 couples (saison, nom) dans deux clubs = transferts en cours de saison (plausibles).
- **Cohérence interne** (comptes complets, pas des exemples) :
  - somme des buts des joueurs ≠ score du match : **587 équipes-matchs** (2007/08 : 91, 2008/09 : 52, 2009/10 : 67, 2010/11 : 121, 2011/12 : 22, 2012/13 : 23, 2013/14 : 41, 2014/15 : 10, 2015/16 : 6, 2016/17 : 1, 2017/18 : 5, 2018/19 : 4, 2019/20 : 2, 2020/21 : 5, 2021/22 : 7, 2022/23 : 12, 2023/24 : 44, 2024/25 : 33, 2025/26 : 33, 2026/27 : 8). Écart de 1 but dans ≈ 80 % des cas récents. **Origine : les PDF de la LNH eux-mêmes** (ex. 011LSL26 : score 25–29 mais total « Buts » de l'équipe = 29 et somme des joueurs = 29) ; avant 2012 ce sont surtout des joueurs manquants dans la feuille.
  - buts > tirs : 6 lignes ; buts 7 m > tirs 7 m : 8 ; arrêts 7 m > tirs 7 m subis : 2 ; tirs 7 m subis > tirs subis : 7 ; minutes > 70 : 1 ; valeurs négatives : 0 ; tirs 7 m > tirs : 0 ; arrêts > tirs subis : 0.
  - temps de jeu cumulé d'une équipe hors [18 000 ; 26 000] s : 44 équipes-matchs ; **le chronomètre (`seconds_played`) est absent à 100 % avant 2023/24** (minutes estimées, signalées par « ~ »).
  - journées : toutes complètes sauf 2020/21 J26 (match annulé, Covid) ; 2019/20 s'arrête à la J18 (Covid). Dates non croissantes (> 14 j) : 33 journées, dont 22 en 2020/21 (reports Covid). 1 match avec mi-temps > score final (`211LSL24`), 6 matchs joués sans mi-temps.
- **Colonnes « non relevées »** (valeur 0 dans le CSV, donc piège) : `br`, `tc`, `pp`, `2p` à 0 avant 2023/24 ; `dp` à 0 avant 2013/14 puis partiel jusqu'en 2022/23 (≈ 6 par match contre ≈ 45 depuis) ; `a` (avertissements) ≈ 5 par match jusqu'en 2018/19 puis ≈ 0,05 en 2023/24 et 0,7 en 2025/26.
- **Ancienne signification de « a »** : recherche dans server, simulateur, index.html, README, exports : plus aucun usage comme passe décisive. `yc` = `a`, `ast` = `dp` (relevé depuis 2023/24). Libellés « LNH » / « dans la base » / `lnh_score` : plus aucune occurrence (hors compatibilité d'une regex de lecture d'ancien cache).
- **Records recalculés indépendamment** (script qui relit les CSV bruts) — 15 records comparés à ce que renvoie le site :

| Record | Calcul indépendant | Site |
|---|---|---|
| Plus de buts sur un match | 19 (B. Richert, 2023/24) | 19 ✓ |
| Plus de buts à 7 m sur un match | 11 (M. Hansen, 2021/22) | 11 ✓ |
| Plus d'arrêts sur un match | 27 (N. Stojinovic, 2007/08) | 27 ✓ |
| Meilleur % d'arrêts (≥ 15 tirs) | 70,0 % (V. Gérard, 14/20) | 70,0 ✓ |
| Plus de passes décisives sur un match (2023+) | 25 | 25 ✓ |
| Plus de buts sur une saison | 241 (T. Pelayo, 2024/25) | 241 ✓ |
| Plus d'arrêts sur une saison | 355 (N. Portner, 2021/22) | 355 ✓ |
| Plus de passes décisives sur une saison | 271 (E. Hanusz, 2025/26) | 271 ✓ |
| Total de buts en carrière | 2 609 (R. Caucheteux) | 2 609 ✓ |
| Matchs disputés | 474 (≥ 1 minute ou ≥ 1 statistique) | 474 ✓ |
| **Plus longue série de matchs avec un but** | Valero Rivera 123, R. Caucheteux 113, N. Ilic 90 | identique ✓ |
| Série de victoires **sur une saison** | 30 (PSG 2021/22) | 30 ✓ |
| Invincibilité sur une saison | 30 (PSG 2021/22) | 30 ✓ |
| Plus large victoire / plus de buts d'une équipe | 50–24 Montpellier à Pau 2012/13 | ✓ |
| Match le plus prolifique | 85 buts (Toulouse–Dijon 2023/24) | 85 ✓ |

  Cas **Nicolas Tournat** : série de 42 matchs (2024/25 → 2026/27), identique en calcul indépendant. Il était à 185 avec l'ancienne méthode, qui enjambait les saisons 2020–2023 passées hors Starligue. Le cas « quitte puis revient » est géré par le calendrier du club (un match manqué ou une saison absente casse la série).

### Phase 2 — Cohérence entre écrans
- **10 joueurs** (Steins, Hansen, Caucheteux, Pelayo, Stojinovic, Gérard, Tournat, Omar, Prandi, Konan) × 11 grandeurs (matchs, buts, tirs, arrêts, BR, TC, BP, 2 min, avertissements, rouges, passes décisives) : CSV brut ↔ fiche joueur ↔ somme des lignes de l'explorateur : **0 écart**.
- **5 clubs** (PSG, Caen — promu 2026/27 —, HBC Nantes, Montpellier HB, Aurillac) sur « toutes saisons » et 2025/26 : matchs, V/N/D, points, buts pour/contre, % de réussite aux tirs — fiche club ↔ carte tactique ↔ calcul indépendant : seuls écarts = arrondis d'affichage (67,35 → 67,3 ou 67,4 ; 28,636 → 28,63/28,64).
- **Classements « Meilleurs buteurs », « Passes décisives », etc. ↔ fiche joueur** : **63 écarts trouvés avant correction**, tous des joueurs ayant changé de club (la liste les coupait en une ligne par club : Butto affichait 1 236 buts, sa fiche 1 436). **0 écart après correction** (15 joueurs × 5 listes × 6 grandeurs).
- **Inversion de la note** : la formule *affichée* dans l'encart, appliquée à la main sur les CSV bruts, a été comparée à `match_note` sur **94 476 lignes joueur-match** : écart maximal **0,0000** (note et note défensive). Un détail du code n'était pas dans l'encart (le terme « contexte d'équipe » est multiplié par 1,65 quand la feuille n'a pas de statistiques défensives) : ajouté. Cohérence entre écrans pour le même match (record « meilleure note » → modale match → explorateur → meilleur match de la fiche) : 9,98 partout ; moyenne de la fiche = moyenne indépendante (écarts ≤ 0,01 hors 2014/15, qui n'est pas appariable dans ce contrôle).
- **Répartition des notes par poste** (joueurs ≥ 10 min, 2023/24 et après, note /10 : moyenne ± écart-type, maximum) :

| Poste | n | Note globale | Note défensive |
|---|---|---|---|
| Demi-centre | 2 789 | 6,69 ± 1,59 (max 9,86) | 5,71 ± 1,49 |
| Arrière gauche | 3 404 | 6,43 ± 1,60 | 6,14 ± 1,82 |
| Arrière droit | 2 604 | 6,39 ± 1,56 | 5,73 ± 1,56 |
| Ailier gauche | 2 658 | 6,33 ± 1,47 | 5,82 ± 1,55 |
| Ailier droit | 2 473 | 6,25 ± 1,44 | 5,68 ± 1,47 |
| Pivot | 3 550 | 6,16 ± 1,39 | 5,82 ± 1,70 |
| **Gardien** | 2 469 | **5,38 ± 1,79** | 5,75 ± 2,07 |

  **Note défensive : équilibrée** entre postes depuis la calibration (écart maximal entre postes 0,46 point). **Note globale : les gardiens sont désavantagés de ≈ 1 point depuis 2023/24** (5,38 contre 6,2–6,7 pour les joueurs de champ), alors qu'ils étaient *avantagés* avant (5,90 contre 5,6–6,1) : les nouvelles colonnes (BR +2,0, TC +1,8, 7 m et 2 min provoqués) ont gonflé la note des joueurs de champ seulement. → décision (d).
- **Carte tactique** : 27 indicateurs × 20 saisons + « toutes saisons » : **aucun pourcentage hors 0–100**, aucun club absent d'une saison. Valeurs manquantes (statistiques non relevées) = `null` des deux côtés ; la note sous le graphique indique désormais combien de clubs sont exclus et pourquoi.
- **Libellés** : « en Starligue depuis 2007 » (« depuis 2023 » pour les passes décisives), « Note /10 » (plus de « Note LNH »), colonnes « Buts / tirs » et « Arrêts / tirs subis » séparées, pluriels testés (0/1/2 point, 0/1/2 match), ordinaux (« 1re victoire », « 2e »).

### Phase 3 — Simulation Monte Carlo
- **Reproductibilité** : même graine ⇒ scores et classements identiques ; graine différente ⇒ différents ; blessures **désactivées par défaut** ; `inj_enabled=True` avec taux 0 ⇒ résultats **identiques** à « désactivé » (le tirage des absences n'utilise pas le générateur des scores).
- **Bon sens** (600 simulations, 6 configurations : 2026/27 à J0, J10, J11 ; 2025/26 à J29 et J30 = saison terminée ; 2023/24 à J5) : rangs = permutation de 1..T dans chaque simulation ; somme des points = 2 × nombre de matchs ; matchs joués jamais re-simulés ; points simulés ≥ points acquis ; matrice des rangs : somme par club et par rang = 1 ; probabilités titre / podium / top 4 / relégation = fréquences de la matrice (écart 0,0). Zéro erreur.
- **Simulateur conditionnel** : sur la simulation affichée (10 000), 1, 2 et 3 verrous donnent **exactement** les mêmes résultats côté navigateur et côté serveur (écart maximal 0) ; combinaison improbable : avertissement clair identique (« seulement 6 simulations… ») ; 0 simulation compatible : message, pas d'erreur. Écart trouvé : le calcul local acceptait 4 verrous ou deux issues du même match (le serveur les refuse) ; aligné.
- **Backtest** 19 saisons × 2 graines (blessures désactivées / activées) : log-loss et Brier 1N2 **inchangés** (0,7513 et 0,4331 à J0 ; les blessures n'agissent que sur la simulation de saison). Score de log du rang réel : 2,1806 → 2,1841 (J0), 1,7409 → 1,7465 (J10) ; probabilité du rang réel 0,141 → 0,139 ; points attendus : erreur +0,07 ± 0,014. **Les blessures dégradent légèrement : elles restent désactivées.**
- **Forces Dixon & Coles** 2026/27 (J10) : signes cohérents (offRTG = buts marqués, defRTG = buts encaissés, net = différence) ; PSG +7,2, Nantes +6,4, … Caen (promu, 9 matchs d'historique) −2,7. La moyenne des offRTG (31,1) dépasse celle des defRTG (30,4) : effet de convexité de l'exponentielle, sans conséquence sur les simulations.
- **Absences** : l'écran distingue bien « constaté à la J10 », « simulé » (malus décroissant de 15 %/journée) et « constaté aujourd'hui ». Le pourcentage affiché est `100·(exp(malus)−1)`, exactement la valeur appliquée au modèle (vérifié dans le code d'export et `apply_availability`). Il n'y a pas de 4ᵉ catégorie « absence aléatoire » puisque les blessures sont désactivées.

### Phase 4 — Fiche journaliste et modale match
- **Fiche journaliste** : les 16 fiches du cache (journée précédente 8 + suivante 8) rendues à **360, 768 et 1 280 px** : aucun « undefined », « NaN », « Invalid Date », « null », aucun débordement horizontal. Dates : tous les formats présents sont `AAAA-MM-JJ` (+ vide) ; affichage en heure locale du navigateur sans décalage ; vide ⇒ « date à confirmer ».
- **Enjeux chiffrés** : recalcul indépendant à partir des fichiers bruts de la simulation (`sim_standings`, `sim_outcomes`) : 8 fiches sur 8 identiques (ex. 23,2 points de pourcentage pour Tremblay sur le top 4 ; 28,2 pour Dunkerque sur le maintien).
- **Modale match** : 101 matchs ouverts (début, fin, un nul, un milieu, un match sans feuille et un match avec disqualification, pour chacune des 20 saisons) : aucune erreur, aucune valeur aberrante, aucun débordement, 287 ms en moyenne (max 1,4 s). Elle n'appelle jamais `/api/simulate`.
- **Face-à-face** : 5 duels (PSG–Montpellier, Aix/PAUC–Nantes, Dunkerque–Nîmes, Saint-Raphaël–Chambéry, Tremblay–Toulouse) : nombre de confrontations, V/N/D et matchs à domicile identiques au calcul indépendant, dans les deux sens et malgré les changements de nom.

### Phase 5 — Site, performance, mémoire, sécurité
- **Parcours** : 31 actions scriptées (3 onglets, frise des saisons, sélecteurs, explorateur, tri, recherche globale, modales club / joueur / match, chips Monte Carlo, fiche journaliste, retour à l'Historique) : 0 erreur JavaScript, 0 promesse rejetée. Échap, piège de tabulation et retour du focus dans les modales : vérifiés.
- **Modes dégradés** : bundle JSON absent ⇒ repli sur l'API (page complète) ; JSON corrompu ⇒ repli sur l'API ; CSV tronqué ⇒ repli sur l'API. Jamais de page blanche.
- **Mémoire** (Windows, RSS / USS) : à vide 108 / 76 Mo ; mode cache (page + bundles) 108 / 76 ; après l'onglet Historique 204 / 163 ; après fiches et modale match 245 / 205 ; **pic pendant une simulation manuelle de 3 000 : 378 / 303 Mo** (404 / 329 avant l'optimisation) ; **génération du cache : 273 / 234 Mo, 21 s** (hors machine à 0,1 CPU).
- **Poids** : `cache/` = 6,1 Mo brut, **2,2 Mo compressé** ; plus gros fichier 1,84 Mo (< 3 Mo) ; chargement initial hors historique ≈ 305 Ko compressé ; `/api/history` 2,26 Mo brut → **232 Ko** une fois compressé (il ne l'était pas).
- **Sécurité** : `/cache/<nom>` (traversées `..`, `%2e%2e`, `%2f`, doubles extensions, `stamp.json`/`status.json`) → 404 ; paramètres de l'explorateur, `q` de 50 Ko, regex, `page_size` énorme, injection dans `seasons`/`team`/`name` → réponses valides ; HTML échappé (test d'injection `<img onerror>` sur la feuille de match, les records, les pastilles de forme : 0 élément injecté).
- **Accessibilité** : contrastes calculés (texte 16,7:1 ; atténué 5,5:1 ; bleu 7,6:1) ; modales `role="dialog" aria-modal aria-labelledby` ; `lang="fr"` ; un seul `h1`.
- **Code mort** : `vulture` + balayage JS/CSS ; fichiers obsolètes dans `cache/`.

---

## (b) Ce qui a été corrigé (avant → après)

| Commit | Correction |
|---|---|
| `ecdb00e` | **Club fantôme** : « ILLEFRANCHE SUR SAÔNE » et « Villefranche sur Saône » étaient deux clubs en 2007/08 (15 clubs, classement et stats faussés) → 14 clubs. Script `tools/audit_data.py`. Nettoyages (import inutilisé, commentaires). |
| `d9c4455` | **Zéros silencieux** : BR, TC, 7 m / 2 min provoqués valaient 0 avant 2023/24, ce qui produisait des « 0 récup. », des cartes tactiques à 0, des notes défensives = contexte d'équipe seul et un 5,0 par défaut. → valeurs **manquantes** (« n.d. », « – », case vide dans l'export), liste « Note défensive » vide avec message explicite avant 2023/24, moyennes calculées sur les seules saisons relevées. `fmt(null)` affiche « – » (et non « 0,0 »). Libellés « Note défensive LNH » → « /10 ». Encart : facteur ×1,65 documenté. |
| `a882486` | Simulateur conditionnel local : mêmes règles que le serveur (1 à 3 verrous, pas de doublon). |
| `a82f73d` | Corps POST limité à 64 Ko (413) ; saison vérifiée et journée bornée (avant : une « saison » bidon lançait un sous-processus `--season=…`, un POST de 5 Mo était lu en mémoire) ; au plus 2 simulations en cours/en attente (429) ; seuls les 20 derniers jobs terminés sont gardés ; bundles périmés de `cache/` supprimés à chaque génération (une simulation lancée depuis le site y déposait des fichiers versionnés). |
| `f1c5eb4` | Focus et piège de tabulation dans les modales ; bouton orange 3,3:1 → 5,2:1 et vert 4,35:1 → 5,4:1 (WCAG AA) ; cibles tactiles ≥ 30 px ; mémoire des feuilles de match libérée avant une simulation manuelle (pic 404 → 378 Mo). |
| `cedfdb6` | Fonction JS `teamsForPeriod` et paramètre `meta_` inutilisés supprimés. |
| `592a73a` (**change des chiffres affichés**, commit séparé) | Les classements joueurs agrègent un joueur sur tous ses clubs (avant : une ligne par club, donc Butto 1 236 buts au lieu de 1 436 et un rang faux) ; l'export CSV n'a plus de colonne `a` ambiguë (remplacée par `yc`) et les passes décisives sont vides — et non 0 — avant 2023/24. |
| `6203e14` | `/api/history` compressé en gzip (2,26 Mo → 232 Ko). |


---

## (c) Ce qui n'a PAS été corrigé, par gravité

**Moyenne**
1. **Incohérences à la source (PDF LNH)** : 587 équipes-matchs où la somme des buts des joueurs ≠ score (surtout avant 2012, mais 33 en 2025/26). Impossible à corriger sans arbitrer entre score et feuille ; les totaux de buts par joueur ne reproduisent donc pas toujours le score. Proposition : badge « feuille incohérente » dans la modale match (non fait).
2. **Notes non comparables entre époques et entre rôles** (voir (d)). Aucune formule modifiée.
3. **Avertissements sous-relevés depuis 2023/24** (0,05 par match contre ≈ 5 avant) : les classements de discipline récents sous-estiment les cartons jaunes. Signalé dans l'encart, pas corrigeable.

**Faible**
4. 6 matchs joués sans date, 6 sans mi-temps, 1 mi-temps > score final (`211LSL24`) : non corrigés (information absente ou erronée à la source) ; les écrans le gèrent (« date à confirmer »).
5. 6 graphiques SVG sans `role="img"`/`aria-label`, pas de lien d'évitement, boutons-liens en ligne de 20 px dans les tableaux (seuil WCAG 2.5.8 : exemption des liens en ligne).
6. `tools/run_pipeline_lnh.py` : 2 f-strings sans variable.
7. Le paramètre `boxscore_df` de `load_all_matches` n'est plus utilisé (conservé par compatibilité).
8. Matrice offRTG/defRTG : moyenne des offRTG ≠ moyenne des defRTG (effet de convexité, sans conséquence).

---

## (d) Décisions qui vous reviennent

1. **Séries de victoires / invincibilité : sur une saison ou toutes saisons ?** Le site mesure par saison (30 pour le PSG en 2021/22). Toutes saisons confondues (en cassant à la sortie de la Starligue) : **39 victoires de suite pour Montpellier** (2010/11 J7 → 2011/12 J19) et **57 matchs sans défaite pour le PSG** (2023/24 → 2024/25), valeurs réelles d'après les CSV. *Recommandation : afficher les deux* (record de saison et record de série), car c'est ce qu'un journaliste cite. Change des chiffres → non fait.
2. **Note /10 : centrage par rôle et par époque.** Depuis 2023/24 la note moyenne des gardiens (5,38) est ≈ 1 point sous celle des joueurs de champ (6,2–6,7) ; avant 2023, la tendance était inverse. *Recommandation* : recentrer le point d'inflexion de la sigmoïde par rôle sur 2023/24+ (≈ 3,0 pour le champ, ≈ 2,0 pour les gardiens), ou comparer chaque joueur à la moyenne de son rôle. Ne touche pas aux classements d'un même rôle.
3. **Note défensive avant 2023/24** : elle n'existe plus pour les joueurs de champ (n.d.). Alternative : l'afficher avec un bandeau « contexte d'équipe seulement ». *Recommandation* : garder « n.d. » (ce que fait le code maintenant).
4. **Passes décisives avant 2023/24** : affichées seulement depuis 2023/24. Les chiffres 2013–2022 (≈ 6 par match) existent mais sont partiels. *Recommandation* : garder.
5. **Limites anti-abus** : corps POST 64 Ko, 2 simulations actives ou en attente, pas de limitation par IP. *Recommandation* : garder ; ajouter une limitation par IP seulement si le site est exposé publiquement avec du trafic.
6. **Mémoire libérée avant une simulation manuelle** : le premier accès à l'Historique qui suit recharge les CSV (quelques secondes ; peut atteindre plusieurs dizaines de secondes à 0,1 CPU). *Recommandation* : garder, c'est ce qui maintient le pic sous 400 Mo.
7. **Couleurs** : orange du bouton `#F25C2A` → `#C2410C`, vert `#168A5C` → `#13795A` (contraste). *Recommandation* : garder ; c'est un choix de charte.

---

## (e) Limites de l'audit

- **Comportement réel sur Render non testé** : mémoire mesurée sous Windows (RSS et USS ; **PSS indisponible sous Windows**, donc non mesuré), pas dans le conteneur de 512 Mo à 0,1 CPU. Les temps (simulation 3 000 en 4 s, génération du cache en 21 s) seront très supérieurs sur Render Free.
- **Impression / PDF de la fiche journaliste** : CSS d'impression présent (menu masqué, fiche seule) mais **non vérifié visuellement** (le navigateur intégré n'imprime pas).
- **Serveur lent et réponse HTTP 500** : non reproduits (aucun endpoint n'a renvoyé 500 sur les entrées testées ; le comportement face à un serveur lent n'a pas été simulé).
- **Journée reportée ou rejouée** dans la simulation : non testée explicitement (les journées incomplètes des CSV sont gérées par le calendrier réel).
- **Captures d'écran** : réalisées à 360 px (fiche journaliste, modale match) ; 768 et 1 280 px contrôlés par mesures du DOM (débordement, valeurs), pas par examen visuel de chaque écran.
- **Contrastes** : calculés sur la palette principale ; les teintes dynamiques (couleurs de clubs, cartes de chaleur) ne l'ont pas été.
- **Workflow GitHub Actions hebdomadaire** : jamais exécuté sur GitHub ; le scraper n'a été lancé qu'en lecture seule du calendrier.
- **Sources** : les statistiques sont celles des PDF de la LNH ; aucune vérification contre une source externe (aucune n'était disponible).
