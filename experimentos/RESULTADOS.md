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

## Segunda etapa: CV 5-fold del nivel 2

Para comparar con menos ruido que un 80/20, se cachean las features de nivel 1 (`cache_l1.py`) y se evalúa el nivel 2 con
5 folds estratificados sobre las 70.000 filas (`l2_cv.py`). Una diferencia de 0,1 punto ≈ 70 filas.

| Experimento (nivel 2) | Accuracy CV |
|---|---|
| Configuración v5: blend 0,3 LightGBM + 0,7 CatBoost | 94,55% |
| CatBoost solo | 94,58% |
| + agregados por vendedor (train+test, sin etiquetas) → **v7** | **94,66%** |
| + vecinos de título dentro del vendedor (etiqueta OOF del ítem más parecido) | sin mejora (LightGBM 94,34% vs 94,32%) |
| CatBoost 1000 iter / lr 0,03 → **v8** | **94,76%** |
| CatBoost 800 iter / lr 0,04 / prof. 8 | 94,74% |
| CatBoost 1000 iter / lr 0,03 / l2 = 10 | 94,74% |
| CatBoost 1600 iter / lr 0,02 | 94,74% |
| CatBoost 1000 iter / lr 0,03 / prof. 6 | 94,61% |
| + BERT v3 y XLM-R (salidas ya existentes) | 94,63% – 94,66% (sin mejora) |
| + BERTIN (RoBERTa en español, Kaggle) | 94,73% |
| + BERTIN + BETO uncased + BETO con otra entrada | 94,72% |
| Regresión logística sobre CatBoost + 5 BERT | 94,79% (+0,03: ruido) |

- **Agregados por vendedor**: mejoran los 5 folds; sobre las mismas filas, 390 aciertos nuevos vs 310 perdidos (McNemar z ≈ 3).
- **BERTs adicionales**: individualmente 89,7% – 90,7% (BETO uncased es el mejor BERT solo), pero no mejoran el ensamble:
  su información (texto del título/atributos) ya está cubierta por los 2 BERT, las LR de TF-IDF y la red neuronal.
  Correlación entre BERTs: 0,87 – 0,95.
- **Fugas descartadas**: los IDs están anonimizados; no hay ítems padre ni fotos compartidas entre train y test; el sufijo
  numérico de la descripción no tiene señal.
- **Descartado por fuga**: promediar las predicciones OOF de los otros ítems del vendedor (los modelos con embedding de
  vendedor ya vieron la etiqueta del propio ítem).
