# Análisis Predictivo – Examen 3 (Kaggle `ap-2026-q-2`)

Clasificación de ítems de MercadoLibre como **nuevos o usados** (métrica: accuracy).

## Modelo final (v7 — Kaggle público 0,94490)
1. **Variables**: numéricas, binarias y de texto del ítem (precio, cantidades, fotos, garantía, medios de pago, fechas, forma del título…).
2. **Stacking (out-of-fold)**: target encoding de vendedor, categoría, ciudad y combinaciones + regresiones logísticas sobre TF-IDF (título, caracteres, atributos).
3. **BERT en español** (`dccuchile/bert-base-spanish-wwm-cased`) ajustado en Colab con 5 folds → probabilidades OOF (`colab/`): v1 (título + categoría + tipo, `colab_bert_oof.py`) y v2 (entrada rica: precio, garantía, atributos, pagos, envío; `colab_bert_v2.py`).
4. **Red neuronal propia** (PyTorch): embeddings de título, categoría, vendedor, etc. + variables numéricas.
5. **Agregados por vendedor** (`src/seller_feats.py`), calculados sobre train + test **sin usar etiquetas**: cantidad de publicaciones, diversidad de categorías, precio medio y desvío del ítem respecto de su vendedor, proporción con garantía, con stock > 1, tipo de publicación, etc.
6. **CatBoost** (2 semillas, 360 iteraciones, lr 0,06) sobre todo lo anterior. Reemplaza al blend LightGBM + CatBoost de la v5: en CV CatBoost solo es mejor.

## Cómo reproducir
```bash
pip install -r requirements.txt
# poner train_data.jsonlines y test_data.jsonlines en data/
python train.py   --train data/train_data.jsonlines --test data/test_data.jsonlines --bert-oof  colab/oof_bert.npy  colab/oof_beto.npy
python predict.py --train data/train_data.jsonlines --test data/test_data.jsonlines --bert-test colab/test_bert.npy colab/test_beto.npy --out submission_v7.csv
```
`--test` en `train.py` (y `--train` en `predict.py`) solo se usa para los agregados por vendedor, que no usan la etiqueta.
Los archivos `colab/oof_*.npy` y `colab/test_*.npy` ya están incluidos (salida de los scripts de Colab/Kaggle).

La otra submission evaluada, **v8** (mejor CV, 94,76%), es la misma con CatBoost más largo: agregar `--cb-iters 1000 --cb-lr 0.03` a `train.py`.

## Submissions
| Submission | Cambio | CV (5 folds) | Kaggle público |
|---|---|---|---|
| C_row1 | stack + LightGBM + CatBoost, sin BERT | — | 0,93966 |
| v4 | + BERT v1 | — | 0,94247 |
| repo | + red neuronal | — | 0,94438 |
| v5 | + BERT v2 (blend 0,3 LGB + 0,7 CB) | 94,55% | 0,94471 |
| v6 | + BERT v3 | — | 0,94452 |
| **v7** ✔ | v5 + agregados por vendedor, CatBoost solo | 94,66% | **0,94490** |
| v8 | v7 con CatBoost 1000 iter / lr 0,03 | **94,76%** | 0,94395 |

Seleccionadas para el leaderboard final: **v7** y **v5**.
La v8 mejora la CV (+0,10) pero no el público: difiere de la v7 en solo 209 filas de test (~146 en el público), así que la diferencia pública (~20 filas) está dentro del ruido.

## Estructura
- `src/`: features, stacking, red neuronal, agregados por vendedor.
- `train.py`, `predict.py`: pipeline final.
- `colab/`: BERT (scripts, notebook `bert_extra.ipynb` para Kaggle/Colab y salidas OOF/test).
- `experimentos/`: anexo con baseline, pruebas descartadas y CV de nivel 2 (`RESULTADOS.md`).
- `presentacion/`: PDF/PPT.
