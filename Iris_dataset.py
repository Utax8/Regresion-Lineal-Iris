from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

'''
Integrantes:
Mateo Beltrán Chavez
Gennier Santiago Caro Alarcón'''

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)   

iris = load_iris()

print(iris.keys())
print(iris.target_names)
print(iris.feature_names)

dataframe_iris = pd.DataFrame(data = iris.data, columns = iris.feature_names)
print(dataframe_iris)

x = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

modelrl = LinearRegression()
modelrl.fit(X_train, y_train)

y_prediccion_cont = modelrl.predict(X_test)

y_predic = np.rint(y_prediccion_cont).astype(int)
y_predic = np.clip(y_predic, 0, 2)

# Precisión del modelo
print("Precision del modelo con regresión lineal:", accuracy_score(y_test, y_predic))
print("Valor de la pendiente:", modelrl.coef_)
print("Valor de la intersección:", modelrl.intercept_)

# Clasificación 
prueba = [[5.1, 3.5, 1.4, 0.2]]
pred = np.rint(modelrl.predict(prueba)).astype(int)[0]
pred = np.clip(pred, 0, 2)
print("La muestra pertenece a la especie", iris.target_names[pred])


# Mostrar los datos Clasificación en dataframe
resultados_test = pd.DataFrame(X_test, columns=iris.feature_names)
resultados_test["Clasificación Real"] = [iris.target_names[i] for i in y_test]
resultados_test["Predicción"] = [iris.target_names[i] for i in y_predic]
print(resultados_test)
