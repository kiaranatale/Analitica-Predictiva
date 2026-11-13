# Experimentos (anexo)

Validación: 80/20 estratificado sobre train (70.000 filas), accuracy. Estos scripts son de referencia; importan módulos
de versiones intermedias del pipeline, por lo que no son ejecutables tal cual contra `src/`.

| Experimento | Accuracy val. |
|---|---|
| Baseline: árbol (prof. 6), 8 variables numéricas | 78,9% |
| Regresión logística, TF-IDF del título | 81,3% |
| LightGBM + variables + TF-IDF | 91,5% |
| + target encoding (vendedor/categoría) + LR de texto apilada | 93,2% |
| CatBoost con categóricas nativas | 93,7% – 94,0% |
| Blend 0,7 CatBoost + 0,3 LightGBM | 93,9% |
| XGBoost | 93,3% |
| LightGBM regularizado (15 hojas) | 93,4% |
| ExtraTrees / MLP sklearn | 92,3% / 92,1% |
| kNN de títulos (palabras/caracteres) | 93,3% (sin mejora) |
| Pseudo-labeling (umbral 0,90 / 0,97) | 93,9% (sin mejora) |
| Optuna (LightGBM y CatBoost) | 93,2% / 93,6% (peor que a mano) |
| BERT español solo (título + categoría) | 87,9% |
| **Final: stack + BERT + red propia → LightGBM + CatBoost** | **~94,1%** |

Leaderboard público (Kaggle): v1 (sin BERT) 0,93966.

Análisis de errores: ~38% de los errores son de confianza media/alta (probable ruido de etiquetado: vendedores que publican
nuevos y usados mezclados); solo 11% se concentra en las 10 categorías con más errores.
