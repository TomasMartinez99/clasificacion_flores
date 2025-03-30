# Clasificador de Especies de Iris

Este proyecto implementa un clasificador de especies de flores Iris utilizando el algoritmo K-Nearest Neighbors (KNN). El dataset de Iris es un conjunto de datos clásico en machine learning que contiene medidas de diferentes especies de flores.

## Descripción del Proyecto

El código realiza las siguientes tareas:

1. **Carga de datos**: Utiliza el dataset de Iris incluido en scikit-learn.
2. **Exploración de datos**: Convierte los datos a un DataFrame de pandas para mejor visualización.
3. **Visualización**: Genera un gráfico de dispersión que muestra la relación entre la longitud y ancho del pétalo, coloreado por especie.
4. **Preparación de datos**: Divide el conjunto de datos en entrenamiento y prueba, y escala las características.
5. **Modelado**: Entrena un clasificador KNN con 5 vecinos.
6. **Evaluación**: Evalúa el modelo utilizando una matriz de confusión y un informe de clasificación.
7. **Predicción**: Implementa una función para predecir la especie de nuevas flores basándose en sus medidas.

## Requisitos

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## Instalación

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Uso

El script principal se puede ejecutar directamente:

```bash
python main.py
```

Para utilizar la función de predicción con nuevas muestras:

```python
# Ejemplo de uso
nueva_flor = [5.1, 3.5, 1.4, 0.2]  # Longitud sépalo, Ancho sépalo, Longitud pétalo, Ancho pétalo
especie_predicha = predecir_flor(*nueva_flor)
print(f"La flor probablemente es de la especie: {especie_predicha}")
```

## Estructura del Proyecto

- `main.py`: Script principal con el código de clasificación
- `grafico_iris.png`: Visualización generada de la distribución de especies

## Resultados

El modelo KNN con 5 vecinos logra una precisión del 100% en el conjunto de prueba, como se muestra en la matriz de confusión:

```
[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]
```

Esta matriz muestra que todas las flores fueron clasificadas correctamente en sus respectivas especies.

## Próximos Pasos

Posibles mejoras para este proyecto:
- Probar diferentes algoritmos de clasificación
- Realizar una búsqueda de hiperparámetros para optimizar el modelo
- Implementar validación cruzada para una evaluación más robusta
- Crear una interfaz de usuario simple para la predicción de especies