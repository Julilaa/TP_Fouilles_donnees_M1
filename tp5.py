import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Charger les données
data = pd.read_csv('./mnt/data/telecom_churn_clean.csv')

# Sélectionner les colonnes nécessaires : 'total_eve_charge', 'total_day_charge', et 'churn'
data = data[['total_eve_charge', 'total_day_charge', 'churn']]
print("Valeurs manquantes avant suppression :\n", data.isnull().sum())

# Supprimer les lignes contenant des NaN
data = data.dropna()
print("Valeurs manquantes après suppression :\n", data.isnull().sum())

# Vérifier les valeurs uniques dans la colonne churn
print("Valeurs uniques avant conversion :", data['churn'].unique())

# Si 'churn' contient déjà 0 et 1, pas besoin de conversion
# Vérifiez la structure des données
print(data.head())

# Séparer les features (X) et la target (y)
X = data[['total_eve_charge', 'total_day_charge']]
y = data['churn']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Créer et entraîner le modèle K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Faire des prédictions
y_pred = knn.predict(X_test)

# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualiser les résultats avec une decision boundary
def plot_decision_boundary(X, y, model, title, step_size=0.5):
    # Limites des axes basées sur les données
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    # Création d'une grille avec une résolution réduite (step_size)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                         np.arange(y_min, y_max, step_size))

    # Prédire sur chaque point de la grille
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Tracer la frontière de décision
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title(title)
    plt.xlabel('Total Evening Charge')
    plt.ylabel('Total Day Charge')
    plt.show()

# Plot decision boundary avec une résolution optimisée
plot_decision_boundary(X_test, y_test, knn, 'K-Nearest Neighbors - Decision Boundary', step_size=0.5)
