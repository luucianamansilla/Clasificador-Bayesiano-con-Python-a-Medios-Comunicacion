from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# -------------------------------
# Enunciado del Ejercicio
# -------------------------------
# Contexto:
# Un medio de comunicación quiere detectar automáticamente si una noticia es falsa ("fake") o verdadera ("real")
# basándose en el contenido textual de la noticia.

# Objetivo:
# Entrenar un clasificador Naive Bayes con un conjunto de noticias etiquetadas, evaluar su rendimiento
# y usarlo para predecir si nuevas noticias son falsas.

# -------------------------------
# Datos simulados
# -------------------------------
data = {
    'text': [
        "El presidente anunció una nueva reforma educativa",
        "Descubren que la vacuna convierte a las personas en robots",
        "La NASA confirma el hallazgo de agua en Marte",
        "Científicos afirman que la Tierra es plana",
        "El ministerio de salud lanza campaña contra el dengue",
        "Celebridades usan crema milagrosa para rejuvenecer 30 años",
        "Se inaugura el nuevo hospital en la ciudad",
        "Estudio revela que comer chocolate cura el cáncer",
        "Gobierno aprueba ley de protección ambiental",
        "Investigadores aseguran que los teléfonos espían nuestros sueños"
    ],
    'label': [
        'real', 'fake', 'real', 'fake', 'real',
        'fake', 'real', 'fake', 'real', 'fake'
    ]
}

df = pd.DataFrame(data)

# -------------------------------
# Preprocesamiento y modelo
# 1-Separar características y etiquetas
# -------------------------------
X = df['text']
y = df['label']

# Convertir texto en vectores de frecuencia
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# 2-Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.3, random_state=42)

# 3-Entrenar el clasificador Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# 4-Evaluación del modelo
# -------------------------------
y_pred = model.predict(X_test)

print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

# -------------------------------
# 5-Clasificación de nuevas noticias
# -------------------------------
nuevas_noticias = [
    "Nuevo estudio demuestra que el café mejora la memoria",
    "Expertos afirman que los gatos pueden hablar con humanos"
]

nuevas_vectorizadas = vectorizer.transform(nuevas_noticias)
predicciones = model.predict(nuevas_vectorizadas)

for noticia, prediccion in zip(nuevas_noticias, predicciones):
    print(f'Noticia: "{noticia}" → Clasificación: {prediccion}')

