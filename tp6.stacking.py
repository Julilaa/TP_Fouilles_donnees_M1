import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Chargement des données
data = pd.read_csv('./mnt/data/wbc.csv')

# Suppression des colonnes inutiles
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Conversion de la colonne 'diagnosis' en valeurs numériques : B -> 0, M -> 1
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Vérification des valeurs manquantes
print("Valeurs manquantes :", data.isnull().sum())

# Séparation des features et de la cible
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Standardisation des données
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Définition des modèles individuels
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier(random_state=1)
lr = LogisticRegression(random_state=1, max_iter=1000)

# Création du StackingClassifier
stacking_clf = StackingClassifier(
    estimators=[
        ('knn', knn),
        ('dt', dt),
        ('lr', lr)
    ],
    final_estimator=LogisticRegression(random_state=1)
)

# Entraîner le StackingClassifier
stacking_clf.fit(X_train, y_train)

# Prédictions et évaluation du StackingClassifier
y_pred_stacking = stacking_clf.predict(X_test)
print("\nRésultats du StackingClassifier :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_stacking):.4f}")
print("Classification Report :\n", classification_report(y_test, y_pred_stacking))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_stacking))

# Analyser les erreurs
# Conversion de X_test en DataFrame pour avoir les colonnes d'origine
X_test_df = pd.DataFrame(X_test, columns=data.drop('diagnosis', axis=1).columns, index=y_test.index)

# Identifier les erreurs (mauvaises prédictions) avec les indices absolus
erreurs_indices = y_test[y_test != y_pred_stacking].index
erreurs = X_test_df.loc[erreurs_indices]

# Ajouter la classe réelle et la classe prédite
erreurs['Classe Réelle'] = y_test.loc[erreurs_indices]
erreurs['Classe Prédite'] = pd.Series(y_pred_stacking, index=y_test.index).loc[erreurs_indices]

# Afficher les erreurs
print("\nErreurs détectées (mauvaises prédictions) :")
print(erreurs)
