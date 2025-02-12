import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb

# Chargement des données
train_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/train.csv")
test_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/test.csv")

# Prétraitement des données
train_target = train_df["smoking"]  # Variable cible
train_df = train_df.drop(["id", "smoking"], axis=1)  # Suppression des colonnes inutiles
test_ids = test_df["id"]  # Sauvegarde des IDs du test set
test_df = test_df.drop("id", axis=1)

# Normalisation des données
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)

# Séparation en ensemble d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(train_df, train_target, test_size=0.2, random_state=42, stratify=train_target)

# Initialisation et entraînement du modèle XGBoost
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    colsample_bytree=0.8,
    subsample=0.8,
    early_stopping_rounds=50,
    random_state=42
)

# Entraînement du modèle
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False 
)

# Évaluation sur le jeu de validation
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_proba)

print(f"AUC-ROC sur le jeu de validation: {roc_auc:.4f}")

# Affichage du nombre optimal d'arbres utilisés
print(f"Nombre optimal d'arbres utilisé : {xgb_model.best_iteration + 1}")

# Courbe ROC
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonale
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Prédictions sur le test set
test_proba = xgb_model.predict_proba(test_df)[:, 1]

# Génération du fichier de soumission
submission_df = pd.DataFrame({'id': test_ids, 'smoking': test_proba})
submission_df.to_csv("sample_submission.csv", index=False)

print(" Fichier de soumission prêt !")
