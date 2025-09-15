# Clasificación de Iris con Regresión Lineal

Este proyecto implementa un modelo de **Machine Learning** para clasificar las flores del dataset **Iris** de sklearn en tres especies (*setosa, versicolor, virginica*) utilizando **Regresión Lineal**.  

El flujo del proyecto incluye:  


## Carga del dataset
Se carga el dataset **Iris** desde la librería `scikit-learn`.  

El dataset contiene:  
- **4 características**: largo/ancho de sépalo y pétalo.  
- **1 etiqueta (y)**: especie de la flor (0 = setosa, 1 = versicolor, 2 = virginica).  

## División de los datos
Se utiliza `train_test_split` de **Scikit-learn** para dividir los datos en:  
- **70% entrenamiento**  
- **30% prueba**  

Esto nos permite entrenar el modelo con un subconjunto de datos y luego evaluarlo en ejemplos que el modelo nunca ha visto

## Predicción
El modelo predice la clase de:  
1. Los datos de prueba (`X_test`).  
2. Nuevas muestras introducidas manualmente.  

Ejemplo:  
```python
sample = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(sample)
