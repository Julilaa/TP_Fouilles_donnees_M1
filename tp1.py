import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger les données
data = pd.read_csv('./mnt/data/auto.csv')

# Afficher les premières lignes pour vérifier le contenu
data.head()

# Sélectionner les colonnes nécessaires : 'mpg' et 'displ'
data = data[['mpg', 'displ']]

# Vérifier les valeurs manquantes
data = data.dropna()

# Séparer les features (X) et la target (y)
X = data[['displ']]
y = data['mpg']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Créer et entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Calculer l'erreur quadratique moyenne (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculer le coefficient de détermination R²
r2 = model.score(X_test, y_test)
print(f"R²: {r2}")

# Visualiser les résultats
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X_test, y_pred, color='red', label='Modèle de régression')
plt.xlabel('Cylindrée (displ)')
plt.ylabel('Miles par gallon (mpg)')
plt.title('Régression linéaire : Cylindrée vs Miles par gallon')
plt.legend()
plt.show()
