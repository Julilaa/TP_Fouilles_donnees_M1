import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Charger les données
data = pd.read_csv('./mnt/data/wbc.csv')

# Sélectionner les colonnes nécessaires : 'concave points_mean', 'radius_mean' et 'diagnosis'
data = data[['concave points_mean', 'radius_mean', 'diagnosis']]
data = data.dropna()

# Convertir la target 'diagnosis' en valeurs numériques (0 et 1)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Séparer les features (X) et la target (y)
X = data[['concave points_mean', 'radius_mean']]
y = data['diagnosis']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Régression logistique
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
print("\nRégression Logistique")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

# Arbre de décision (Gini)
dt_gini = DecisionTreeClassifier(criterion='gini', random_state=1)
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
print("\nArbre de Décision (Gini)")
print("Accuracy:", accuracy_score(y_test, y_pred_gini))
print("\nClassification Report:\n", classification_report(y_test, y_pred_gini))

# Arbre de décision (Entropy)
dt_entropy = DecisionTreeClassifier(criterion='entropy', random_state=1)
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
print("\nArbre de Décision (Entropy)")
print("Accuracy:", accuracy_score(y_test, y_pred_entropy))
print("\nClassification Report:\n", classification_report(y_test, y_pred_entropy))

# Visualisation des boundaries
def plot_decision_boundary(X, y, model, title):
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = model.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))
    plt.title(title)
    plt.xlabel('concave points_mean')
    plt.ylabel('radius_mean')
    plt.show()

# Plot decision boundaries
plot_decision_boundary(X_test, y_test, log_reg, 'Régression Logistique - Decision Boundary')
plot_decision_boundary(X_test, y_test, dt_gini, 'Arbre de Décision (Gini) - Decision Boundary')
plot_decision_boundary(X_test, y_test, dt_entropy, 'Arbre de Décision (Entropy) - Decision Boundary')

# Visualiser l'arbre de décision
plt.figure(figsize=(15, 10))
plot_tree(dt_gini, feature_names=['concave points_mean', 'radius_mean'], class_names=['B', 'M'], filled=True)
plt.title("Arbre de Décision (Gini)")
plt.show()

plt.figure(figsize=(15, 10))
plot_tree(dt_entropy, feature_names=['concave points_mean', 'radius_mean'], class_names=['B', 'M'], filled=True)
plt.title("Arbre de Décision (Entropy)")
plt.show()
