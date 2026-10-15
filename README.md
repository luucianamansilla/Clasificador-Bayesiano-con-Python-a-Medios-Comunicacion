# Clasificador Bayesiano de Noticias Falsas

Ejercicio de **Procesamiento de Aprendizaje Automático** — Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial (Instituto Tecnológico Beltrán).

**Alumna:** Angeles Mansilla
**Profesora:** Yanina Scudero

---

## Descripción

Un medio de comunicación necesita un sistema que clasifique noticias como **falsas (`fake`)** o **verdaderas (`real`)** usando solo su texto. Este ejercicio entrena un clasificador **Naive Bayes multinomial** sobre un conjunto pequeño de titulares etiquetados, evalúa el modelo y lo usa para predecir noticias nuevas.

## Tecnologías

- Python 3
- pandas
- scikit-learn: `CountVectorizer`, `MultinomialNB`, `classification_report`, `confusion_matrix`

## Estructura

```
.
├── README.md
└── codigo_BAYES2.py   # Script con la solución completa
```

## Cómo funciona

1. **Datos:** DataFrame con 10 titulares en español (5 `real`, 5 `fake`).
2. **Vectorización:** `CountVectorizer` convierte cada titular en un vector de frecuencia de palabras (Bag of Words).
3. **Entrenamiento:** `MultinomialNB` se ajusta con el 100 % de los datos.
4. **Evaluación:** se predice sobre los mismos datos de entrenamiento y se imprimen el reporte de clasificación y la matriz de confusión.
5. **Predicción:** se clasifican dos noticias que el modelo no vio durante el entrenamiento.

## Ejecución

```bash
pip install pandas scikit-learn
python codigo_BAYES2.py
```

## Resultados

Salida obtenida al ejecutar el script:

```
--- REPORTE DEL MODELO ---
Reporte de clasificación:
              precision    recall  f1-score   support

        fake       1.00      1.00      1.00         5
        real       1.00      1.00      1.00         5

    accuracy                           1.00        10
   macro avg       1.00      1.00      1.00        10
weighted avg       1.00      1.00      1.00        10

Matriz de confusión:
[[5 0]
 [0 5]]

--- CLASIFICACIÓN DE NUEVOS MENSAJES ---
Noticia: "Nuevo estudio demuestra que el café mejora la memoria" → Clasificación: fake
Noticia: "Expertos afirman que los gatos pueden hablar con humanos" → Clasificación: fake
```

**Matriz de confusión** (filas: clase real; columnas: clase predicha; orden `real`, `fake`):

|               | Predicho `real` | Predicho `fake` |
| ------------- | :-------------: | :-------------: |
| **Real**      | 5               | 0               |
| **Fake**      | 0               | 5               |

### Interpretación

- **Accuracy 100 %** sobre los datos de entrenamiento: sin errores. Es un resultado esperable porque el modelo se evalúa con los mismos datos con los que se entrenó y el dataset es muy chico; **no indica capacidad de generalización**.
- Ambas noticias nuevas se clasificaron como `fake`. Una explicación probable es el vocabulario compartido con titulares falsos del entrenamiento: "estudio" aparece en *"Estudio revela que comer chocolate cura el cáncer"* y "afirman" en *"Científicos afirman que la Tierra es plana"*. El modelo aprende palabras asociadas a cada clase, no la veracidad del contenido.


**Mejoras posibles:** dataset más grande y variado, `train_test_split` o validación cruzada, TF-IDF en lugar de conteos simples y limpieza de texto (stopwords, normalización).
