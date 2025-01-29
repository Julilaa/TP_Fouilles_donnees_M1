import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


# Wrapper pour XGBClassifier avec gestion des classes
class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_  # Définir l'attribut classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


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

# Optimisation des modèles de base
dt = DecisionTreeClassifier(max_depth=10, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
rf = RandomForestClassifier(n_estimators=200, random_state=1)
xgb = XGBWrapper(n_estimators=100, random_state=1, eval_metric='logloss', use_label_encoder=False)

# Recherche des meilleurs hyperparamètres pour la Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(LogisticRegression(random_state=1, max_iter=1000), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Meilleurs paramètres
best_params = grid_search.best_params_
print(f"Meilleurs hyperparamètres pour Logistic Regression : {best_params}")

# Utilisation des meilleurs hyperparamètres dans le StackingClassifier
optimized_lr = LogisticRegression(random_state=1, max_iter=1000, **best_params)

stacking_clf = StackingClassifier(
    estimators=[
        ('knn', knn),
        ('dt', dt),
        ('rf', rf),
        ('xgb', xgb)
    ],
    final_estimator=optimized_lr
)

# Entraîner et évaluer le modèle
stacking_clf.fit(X_train, y_train)
y_pred_stacking = stacking_clf.predict(X_test)

print("\nRésultats du StackingClassifier avec XGBClassifier (Wrapper) :")
print(f"Accuracy : {accuracy_score(y_test, y_pred_stacking):.4f}")
print("Classification Report :\n", classification_report(y_test, y_pred_stacking))
print("Confusion Matrix :\n", confusion_matrix(y_test, y_pred_stacking))

# Analyser les erreurs
X_test_df = pd.DataFrame(X_test, columns=data.drop('diagnosis', axis=1).columns, index=y_test.index)
erreurs_indices = y_test[y_test != y_pred_stacking].index
erreurs = X_test_df.loc[erreurs_indices]
erreurs['Classe Réelle'] = y_test.loc[erreurs_indices]
erreurs['Classe Prédite'] = pd.Series(y_pred_stacking, index=y_test.index).loc[erreurs_indices]

print("\nErreurs détectées après ajout du XGBClassifier :")
print(erreurs)
