# Análisis Predictivo – Examen 3 (Kaggle `ap-2026-q-2`)

Clasificación de ítems de MercadoLibre como **nuevos o usados** (métrica: accuracy).

## Modelo final
1. **Variables**: numéricas, binarias y de texto del ítem (precio, cantidades, fotos, garantía, medios de pago, fechas, forma del título…).
2. **Stacking (out-of-fold)**: target encoding de vendedor, categoría, ciudad y combinaciones + regresiones logísticas sobre TF-IDF (título, caracteres, atributos).
3. **BERT en español** (`dccuchile/bert-base-spanish-wwm-cased`) ajustado sobre título + categoría + tipo de publicación en Colab, 5 folds → probabilidades OOF (`colab/`).
4. **Red neuronal propia** (PyTorch): embeddings de título, categoría, vendedor, etc. + variables numéricas.
5. **LightGBM + CatBoost** (2 semillas) sobre todo lo anterior. Blend final: 0,3 LightGBM + 0,7 CatBoost.

## Cómo reproducir
```bash
pip install -r requirements.txt
# poner train_data.jsonlines y test_data.jsonlines en data/
# (opcional) regenerar colab/oof_bert.npy y colab/test_bert.npy con colab/colab_bert_oof.py en Colab (GPU T4)
python train.py   --train data/train_data.jsonlines --bert-oof colab/oof_bert.npy   # entrena y guarda models/
python predict.py --test  data/test_data.jsonlines  --bert-test colab/test_bert.npy # genera submission.csv
```
Los archivos `colab/oof_bert.npy` y `colab/test_bert.npy` ya están incluidos (salida de `colab_bert_oof.py`).

## Estructura
- `src/`: features, stacking, red neuronal.
- `train.py`, `predict.py`: pipeline final.
- `colab/`: BERT (script y salidas).
- `experimentos/`: anexo con baseline y pruebas descartadas (`RESULTADOS.md`).
- `presentacion/`: PDF/PPT.
