# Guion — 5 minutos

Ritmo: ~130 palabras por minuto. Los tiempos son acumulados. El anexo (lámina 9) **no se presenta**: es para las preguntas.

---

### Lámina 1 · Título — 0:00 → 0:10
> Buenas. Voy a presentar mi modelo para la competencia de Kaggle: predecir si un ítem de MercadoLibre es nuevo o usado.

### Lámina 2 · Datos y métrica — 0:10 → 0:35
> Tenemos 70.000 ítems de entrenamiento y 30.000 de test. Las clases están casi balanceadas: 53,7% son nuevos, así que predecir siempre "nuevo" ya da 53,7%. Ese es el piso.
> La métrica es accuracy. Para validar usé primero un 80/20 estratificado y, al final, validación cruzada de 5 folds sobre todo el train, porque las mejoras eran de décimas y con 14.000 filas no se distinguen del ruido.

### Lámina 3 · Baseline — 0:35 → 1:10
> Como baseline entrené un árbol de decisión de profundidad 6 con 8 variables numéricas, como precio y cantidades: 78,9%.
> Una regresión logística solo con el TF-IDF del título ya da 81,3%: el texto es la señal principal. Por ejemplo, palabras como "usado" o "nuevo sin uso".
> Combinando las dos cosas, variables y texto, en un LightGBM llegué a 91,5%. Fue el salto más grande de todo el proyecto.

### Lámina 4 · Selección de modelos, etapa 1 — 1:10 → 2:00
> A partir de ahí fui sumando piezas y midiendo cada una.
> Lo primero fue el vendedor: un vendedor casi siempre vende solo nuevos o solo usados. Lo codifiqué con target encoding, es decir, la proporción de usados de cada vendedor, calculada out-of-fold para no filtrar la etiqueta. Junto con regresiones logísticas de texto apiladas: 93,2%.
> CatBoost resultó el mejor modelo individual, porque maneja nativamente las categóricas de muchísimos valores, como el vendedor o la ciudad.
> Después sumé dos modelos de texto más potentes: BETO, un BERT en español ajustado con 5 folds, y una red neuronal propia. Llegué a 94,1%.
> XGBoost, ExtraTrees y el ajuste con Optuna no mejoraron.

### Lámina 5 · Selección de modelos, etapa 2 — 2:00 → 2:55
> En la segunda etapa ya trabajé con CV de 5 folds.
> Primero vi que CatBoost solo era igual o mejor que el blend con LightGBM, así que simplifiqué el modelo.
> La mejora que se quedó fueron los agregados por vendedor: cuántas publicaciones tiene, en cuántas categorías vende, si el precio del ítem está por arriba o por debajo de su promedio. Se calculan con train y test juntos, pero sin usar la etiqueta. Mejoró en los 5 folds.
> También probé entrenar CatBoost más tiempo, que subió la CV pero bajó en el público, y entrené tres BERT nuevos. Solos rinden 90–91%, pero no suman al ensamble: repiten la información del texto que ya tenía.

### Lámina 6 · Modelo final — 2:55 → 3:45
> El modelo final es un stacking en dos niveles.
> En el nivel 1 están el target encoding, las regresiones logísticas sobre TF-IDF, la red neuronal, los dos BETO y los agregados por vendedor. Todo lo que usa la etiqueta se calcula out-of-fold: cada ítem se predice con un modelo que no lo vio. Si no, el nivel 2 aprendería a confiar en predicciones que en test no van a ser tan buenas.
> En el nivel 2, CatBoost con dos semillas toma las variables originales más esas salidas. La predicción es el promedio de los dos con umbral 0,5.

### Lámina 7 · Resultados — 3:45 → 4:10
> En el leaderboard público pasé de 0,93966 en la primera entrega a 0,94490 con este modelo, medio punto. Elegí como finales la v7, que es la mejor en el público y mejora la CV respecto de la v5, y la v5 como respaldo.

### Lámina 8 · Limitaciones y mejoras — 4:10 → 4:50
> ¿Dónde está el error que queda? Sobre todo en vendedores que no aparecen en train: son el 30% de las filas, pero casi el 60% de los errores: ahí acierta 89,8% contra 96,8% con vendedores conocidos, porque pierde su mejor pista.
> Además, cerca del 39% de los errores se cometen con más de 80% de confianza, lo que sugiere etiquetas ruidosas: vendedores que cargan mal la condición.
> El texto está agotado: más BERT no suma. Lo que mejoraría el modelo es información nueva sobre el vendedor, como historial o reputación, o un modelo específico para vendedores sin historial.
> Gracias.

---

# Preguntas probables

**¿Por qué out-of-fold?** Si calculo el target encoding o las predicciones de BERT con el mismo dato que después uso para entrenar el nivel 2, ese dato "ya conoce" su etiqueta. El nivel 2 confiaría demasiado en esas columnas y en test fallaría. Con 5 folds, cada fila recibe la predicción de un modelo entrenado sin ella.

**¿Cómo funciona el target encoding?** Para cada vendedor (y para combinaciones como vendedor-categoría): `(usados + 2·prior) / (n + 2)`. El 2 suaviza: con un vendedor de 1 ítem, el valor queda cerca del promedio global. Además, como feature, va la cantidad de ítems del grupo.

**Los agregados usan el test, ¿no es trampa?** No uso la etiqueta del test, solo sus variables (cuántas publicaciones tiene el vendedor, sus precios). Es información que se tiene al momento de predecir. Lo que sí descarté fue promediar las predicciones de los otros ítems del vendedor, porque filtraba la etiqueta propia a través de modelos que la habían visto.

**¿Por qué CatBoost y no LightGBM?** CatBoost codifica las categóricas con estadísticas ordenadas (una especie de target encoding sin fuga) y usa árboles simétricos, que regularizan bien. Con vendedor, ciudad y combinaciones de miles de valores rindió más. En CV: CatBoost solo 94,58% vs blend 94,55%.

**¿Cómo sabés que la mejora de los agregados es real?** Mejora los 5 folds. Comparando fila por fila: 390 aciertos nuevos contra 310 perdidos. El test de McNemar mira solo esas filas que cambian: (390 − 310) / √(390 + 310) ≈ 3, significativo.

**¿Por qué no elegiste la v8 si tiene mejor CV?** Mejora la CV en 0,10, pero bajó en el público. Difiere de la v7 en solo 209 filas; en el público son ~146, y la diferencia es de ~20 filas: no es significativa en ninguna dirección. Me quedé con la v7, que es mejor que la v5 en las dos medidas.

**¿Por qué los BERT nuevos no suman?** Correlacionan 0,87–0,95 con los BERT que ya estaban. Cuando dos modelos se equivocan en los mismos ítems, combinarlos no corrige nada. La información del título ya estaba cubierta por los BETO, las regresiones sobre TF-IDF y la red.

**¿Qué hace la red neuronal?** Promedia embeddings de las palabras del título (más bigramas con hashing) y de los atributos, suma embeddings para categoría, vendedor, ciudad, etc., y las variables numéricas normalizadas. Todo pasa por un MLP de 256 → 64 → 1 con dropout. Se entrena 7 épocas. Sola da 91,7%.

**¿Cómo se entrenaron los BERT?** Fine-tuning de BETO (`bert-base-spanish-wwm-cased`) en Colab/Kaggle con GPU, 5 folds. Para train guardo la predicción out-of-fold; para test, el promedio de los 5 modelos. v1: título + categoría + tipo de publicación. v2: además precio, garantía, atributos, pagos y envío, 3 épocas.

**¿Es reproducible?** Sí: `train.py` y `predict.py` con semillas fijas. Las salidas de BERT vienen precalculadas en `colab/` porque requieren GPU; los scripts para regenerarlas también están.

**¿Qué es ese 39% de errores con confianza alta?** El modelo da más de 0,8 (o menos de 0,2) y aun así se equivoca. Medido sobre las predicciones out-of-fold de la v7. Suelen ser ítems de vendedores que publican nuevos y usados mezclados, o con la condición mal cargada: ruido de etiquetado que ningún modelo puede corregir.

# Números para tener a mano
- Piso: 53,7% · Baseline árbol: 78,9% · LR título: 81,3% · LightGBM: 91,5%
- CV final v7: 94,66% · Público v7: 0,94490 · Primera entrega: 0,93966
- Vendedor de test visto en train: ~72% de las filas · En CV: vendedores no vistos = 31% de filas, 59% de errores
