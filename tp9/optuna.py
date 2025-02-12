# 📌 Importation des bibliothèques
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb

# 📌 Désactiver les logs inutiles
optuna.logging.set_verbosity(optuna.logging.WARNING)  # 🔇 Désactiver les logs Optuna
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # 🔇 Supprimer les UserWarnings XGBoost

# 📌 Chargement des données
train_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/train.csv")
test_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/test.csv")

# 📌 Prétraitement des données
train_target = train_df["smoking"]
train_df = train_df.drop(["id", "smoking"], axis=1)
test_ids = test_df["id"]
test_df = test_df.drop("id", axis=1)

# 📌 Normalisation des données
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)

# 📌 Séparation en ensemble d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df, train_target, test_size=0.2, random_state=42, stratify=train_target
)

# 📌 Fonction d'optimisation des hyperparamètres avec Optuna
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,  # 🔇 Supprime les logs XGBoost
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
        "random_state": 42,
        "early_stopping_rounds": 50  # ✅ Ajout de l'early stopping
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False  # 🔇 Supprime les logs d'entraînement
    )

    y_val_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_val_proba)

# 📌 Lancement de l'optimisation avec 30 essais
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# 📌 Meilleurs hyperparamètres trouvés
best_params = study.best_params
best_params["early_stopping_rounds"] = 50
best_params["verbosity"] = 0  # 🔇 Supprime les logs XGBoost
print("🔹 Meilleurs paramètres trouvés :", best_params)

# 📌 Entraînement du modèle final avec les meilleurs hyperparamètres
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False  # 🔇 Supprime les logs d'entraînement
)

# 📌 Évaluation finale
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_proba)
print(f"🚀 AUC-ROC optimisé : {roc_auc:.4f}")

# ✅ Vérification si `best_iteration` est disponible
if hasattr(xgb_model, "best_iteration"):
    print(f"🔹 Nombre optimal d'arbres utilisé : {xgb_model.best_iteration + 1}")
else:
    print("⚠️ `best_iteration` non disponible, l'entraînement a atteint `n_estimators`.")

# 📌 Courbe ROC
fpr, tpr, _ = roc_curve(y_val, y_val_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 📌 Prédictions sur le test set
test_proba = xgb_model.predict_proba(test_df)[:, 1]

# 📌 Génération du fichier de soumission
submission_df = pd.DataFrame({'id': test_ids, 'smoking': test_proba})
submission_df.to_csv("sample_submission.csv", index=False)

print("✅ Fichier de soumission prêt !")
