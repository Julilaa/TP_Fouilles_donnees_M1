import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger les données
data = pd.read_csv('./mnt/data/auto.csv')

# Sélectionner les colonnes nécessaires : 'mpg' et 'displ'
data = data[['mpg', 'displ']]
data = data.dropna()

# Séparer les features (X) et la target (y)
X = data[['displ']]
y = data['mpg']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Régression linéaire
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print("\nRégression Linéaire")
print(f"MSE: {mse_lin}")

# Arbre de décision régressif
dt_reg = DecisionTreeRegressor(random_state=123)
dt_reg.fit(X_train, y_train)
y_pred_tree = dt_reg.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print("\nArbre de Décision (Régression)")
print(f"MSE: {mse_tree}")

# Validation croisée avec K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=123)
scores_lin = cross_val_score(lin_reg, X, y, scoring='neg_mean_squared_error', cv=kf)
scores_tree = cross_val_score(dt_reg, X, y, scoring='neg_mean_squared_error', cv=kf)

# Moyenne des scores de validation croisée
print("\nValidation Croisée (K-Fold, 5 splits)")
print(f"Régression Linéaire - MSE moyen: {-np.mean(scores_lin)}")
print(f"Arbre de Décision - MSE moyen: {-np.mean(scores_tree)}")

# Visualiser les résultats (Régression linéaire)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Données réelles')
plt.plot(X_test, y_pred_lin, color='red', label='Régression Linéaire')
plt.xlabel('Cylindrée (displ)')
plt.ylabel('Miles per gallon (mpg)')
plt.title('Régression Linéaire : Cylindrée vs Miles per gallon')
plt.legend()
plt.show()

# Visualiser l'arbre de décision
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Données réelles')
plt.scatter(X_test, y_pred_tree, color='green', label='Arbre de Décision (Régression)')
plt.xlabel('Cylindrée (displ)')
plt.ylabel('Miles per gallon (mpg)')
plt.title('Arbre de Décision (Régression) : Cylindrée vs Miles per gallon')
plt.legend()
plt.show()

# Visualiser l'arbre de décision
plt.figure(figsize=(15, 10))
plot_tree(dt_reg, feature_names=['displ'], filled=True)
plt.title("Arbre de Décision (Régressif)")
plt.show()
