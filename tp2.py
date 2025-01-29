import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Charger les données
data = pd.read_csv('./mnt/data/wbc.csv')

# Afficher les premières lignes pour vérifier le contenu
data.head()

# Sélectionner les colonnes nécessaires : 'concave points_mean', 'radius_mean' et 'diagnosis'
data = data[['concave points_mean', 'radius_mean', 'diagnosis']]

# Vérifier les valeurs manquantes
data = data.dropna()

# Convertir la target 'diagnosis' en valeurs numériques (0 et 1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Séparer les features (X) et la target (y)
X = data[['concave points_mean', 'radius_mean']]
y = data['diagnosis']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Faire des prédictions sur les données de test
y_pred = model.predict(X_test)

# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Visualiser les résultats avec une matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion')
plt.colorbar()
plt.xticks([0, 1], ['B', 'M'])
plt.yticks([0, 1], ['B', 'M'])
plt.xlabel('Prédictions')
plt.ylabel('Vérités terrain')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')
plt.show()
