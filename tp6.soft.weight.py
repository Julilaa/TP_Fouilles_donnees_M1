import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
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

# Création et entraînement du VotingClassifier avec des poids
voting_clf_weighted = VotingClassifier(
    estimators=[
        ('knn', knn),
        ('dt', dt),
        ('lr', lr)
    ],
    voting='soft',
    weights=[1, 1, 2]  # Poids : KNN=1, Decision Tree=1, Logistic Regression=2
)
voting_clf_weighted.fit(X_train, y_train)

# Prédictions et évaluation du VotingClassifier pondéré
y_pred_voting_weighted = voting_clf_weighted.predict(X_test)
print("\nRésultats du VotingClassifier (Soft Voting avec poids) :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_voting_weighted):.4f}")
print("Classification Report :\n", classification_report(y_test, y_pred_voting_weighted))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_voting_weighted))
