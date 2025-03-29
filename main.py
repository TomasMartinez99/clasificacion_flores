import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
# Herramientas de ScikitLearn.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Dataset de iris
iris = load_iris()

# Separar los datos en características (X) y etiquetas (y).
X = iris.data
y = iris.target

# Guardar los nombres de las características y especies para usarlos después.
nombres_caract = iris.feature_names
nombres_especies = iris.target_names

# Convertir a DataFrame para visualizarlo mejor.
df = pd.DataFrame(X, columns=nombres_caract)

# Crear una lista con los nombres de las especies y añadirla como columna al DataFrame.
especies = []
for i in y:
    especies.append(nombres_especies[i])
df['especie'] = especies

# Hacer un gráfico para ver como se distribuyen las especies.
plt.figure(figsize=(10, 6))

# Este gráfico mostrará la relación entre la longitud del pétalo (eje X) 
# y el ancho del pétalo (eje Y) para todas las flores del dataset.
sns.scatterplot(data=df, x=nombres_caract[2], y=nombres_caract[3], hue='especie')

# Dividir los datos en entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos para que funcionen mejor con el modelo.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entreno un modelo KNN que es simple pero efectivo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

# Evalúo qué tan bien funcionó mi modelo
print("\nResultados del modelo:")
print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred, target_names=nombres_especies))

# Función para predecir nuevas flores
def predecir_flor(longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo):
    """Predice la especie de una flor iris basada en sus medidas"""
    nueva_flor = np.array([[longitud_sepalo, ancho_sepalo, longitud_petalo, ancho_petalo]])
    nueva_flor_scaled = scaler.transform(nueva_flor)
    prediccion = knn.predict(nueva_flor_scaled)[0]
    return nombres_especies[prediccion]

# Ejemplo de uso con una flor de prueba
nueva_flor = [5.1, 3.5, 1.4, 0.2]
print("\nPrueba con nueva flor:")
print(f"Medidas: {nueva_flor}")
print(f"Predicción: {predecir_flor(*nueva_flor)}")