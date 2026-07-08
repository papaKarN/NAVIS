# NAVIS

**NA**nopore **V**isualization & **I**nteractive **S**tatistiques

NAVIS est un outil de contrôle qualité (QC) pour données de séquençage Oxford Nanopore. Il prend en entrée des fichiers de statistiques par read (générés par [nano_extract](https://github.com/papaKarN/nano_extract)) et produit un **rapport HTML interactif unique**, autonome et partageable, combinant tableaux, histogrammes, courbes de distribution et heatmaps.

## Fonctionnalités

- **Analyse multi-fichiers** : comparez plusieurs runs/échantillons dans un seul rapport
- **Tableau de statistiques cumulées** (Raw / Filtered) : nombre de reads, bases totales, longueur moyenne/médiane/min/max, N50, percentiles de longueur, qualité moyenne/médiane/percentiles
- **Histogrammes résumés** (4 panneaux, barres horizontales)
- **Courbes de distribution** (longueur, bases, qualité) avec 3 modes d'affichage : Raw / Filtered / No Outliers
- **Heatmaps longueur vs qualité** (stacked), avec les mêmes 3 modes
- **Détection d'outliers** configurable via percentile de longueur (`--outlier_percentile`, défaut 99.5)
- **Filtres qualité/longueur** appliqués en direct (`--min_len`, `--max_len`, `--min_qual`, `--max_qual`)
- **Exports TXT** des statistiques (brutes et filtrées)
- Rapport HTML **minifié**, avec version **.html.gz** générée automatiquement (s'ouvre nativement dans Firefox/Chrome, idéal pour le partage)
- Support des fichiers `.gz` (décompression via `pigz` si disponible, sinon `gzip`)
- Mode `--low_memory` pour les très gros jeux de données (traitement séquentiel, libération mémoire au fur et à mesure)
- Mode `--light_html` pour réduire la taille du rapport (moins de bins sur les heatmaps)

## Installation

```bash
git clone https://github.com/papaKarN/NAVIS.git
cd NAVIS
pip install -r requirements.txt
```

Python 3.8+ recommandé. Optionnel mais recommandé pour les gros fichiers `.gz` : installer [`pigz`](https://zlib.net/pigz/) (décompression parallèle).

```bash
# Debian/Ubuntu
sudo apt install pigz
```

## Utilisation

```bash
python NAVIS.py -i sample1.txt sample2.txt -o rapport.html
```

### Arguments

| Argument | Description | Défaut |
|---|---|---|
| `-i`, `--input` | Un ou plusieurs fichiers TXT en entrée (requis) | — |
| `-o`, `--output` | Fichier HTML de sortie | `nanopore_multi_summary.html` |
| `-t`, `--threads` | Nombre de threads CPU | 0 (tous les CPU disponibles) |
| `-b`, `--bin_size` | Taille des bins (bp) pour les heatmaps | 1000 |
| `--min_len` | Longueur minimale de read (bp) | aucun |
| `--max_len` | Longueur maximale de read (bp) | aucun |
| `--min_qual` | Qualité moyenne minimale | aucun |
| `--max_qual` | Qualité moyenne maximale | aucun |
| `--outlier_percentile` | Percentile de longueur au-delà duquel un read est un outlier | 99.5 |
| `--low_memory` | Traitement séquentiel, économe en RAM | désactivé |
| `--light_html` | Réduit la résolution des heatmaps pour un fichier plus léger | désactivé |

### Exemple complet

```bash
python NAVIS.py \
  -i run1.txt run2.txt run3.txt \
  -o rapport_qc.html \
  -t 4 \
  -b 500 \
  --min_len 500 --max_len 30000 \
  --min_qual 8 --max_qual 25 \
  --outlier_percentile 99
```

## Format des données d'entrée

NAVIS attend des fichiers **TSV avec en-tête**, tels que produits par [nano_extract](https://github.com/papaKarN/nano_extract) :

```
read_id     length    mean_quality
read_001    5432      14.3
read_002    3021      12.8
...
```

- `read_id` : identifiant du read (non utilisé pour les calculs, colonne conservée pour compatibilité)
- `length` : longueur du read en bp
- `mean_quality` : qualité moyenne du read

Les fichiers `.gz` sont supportés automatiquement (extension `.txt.gz`).

## Sorties générées

Pour une commande `-o rapport.html`, NAVIS génère :

- `rapport.html` — rapport interactif (minifié)
- `rapport.html.gz` — version compressée du même rapport
- statistiques brutes et filtrées au format `.txt`

## Pipeline recommandé

```
Fichiers Nanopore (FASTQ) 
        │
        ▼
  nano_extract  →  fichiers .txt (read_id, length, mean_quality)
        │
        ▼
     NAVIS  →  rapport HTML interactif
```

## Licence

Ce projet est distribué sous licence MIT — voir [LICENSE](LICENSE).

## Auteur

[papaKarN](https://github.com/papaKarN)
