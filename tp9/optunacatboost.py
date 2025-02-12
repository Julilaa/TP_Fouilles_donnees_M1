#  Importation des bibliothèques
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
import xgboost as xgb
import catboost as cb

#  Désactiver les logs inutiles
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#  Chargement des données
train_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/train.csv")
test_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/test.csv")

#  Prétraitement des données
train_target = train_df["smoking"]
train_df = train_df.drop(["id", "smoking"], axis=1)
test_ids = test_df["id"]
test_df = test_df.drop("id", axis=1)

#  Normalisation des données
scaler = StandardScaler()
train_df = scaler.fit_transform(train_df)
test_df = scaler.transform(test_df)

#  Séparation en ensemble d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(
    train_df, train_target, test_size=0.2, random_state=42, stratify=train_target
)

#  Fonction d'optimisation des hyperparamètres avec Optuna
def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "verbosity": 0,
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.001, 10, log=True),
        "random_state": 42,
        "early_stopping_rounds": 50
    }

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    y_val_proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_val_proba)

#  Augmenter `n_trials` pour une meilleure recherche
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

#  Meilleurs hyperparamètres trouvés
best_params = study.best_params
best_params["early_stopping_rounds"] = 50
best_params["verbosity"] = 0
print(" Meilleurs paramètres trouvés :", best_params)

#  Entraînement du modèle final avec les meilleurs hyperparamètres
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

#  Évaluation finale
y_val_proba = xgb_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_val_proba)
print(f" AUC-ROC optimisé : {roc_auc:.4f}")

#  Vérification `best_iteration`
if hasattr(xgb_model, "best_iteration"):
    print(f" Nombre optimal d'arbres utilisé : {xgb_model.best_iteration + 1}")

#  Feature Importance (pour comprendre les variables clés)
plt.figure(figsize=(10, 6))
xgb.plot_importance(xgb_model, max_num_features=15)
plt.title("Top 15 Features les plus importantes")
plt.show()

#  Test avec CatBoost pour comparaison
cat_model = cb.CatBoostClassifier(
    iterations=1000,
    learning_rate=0.01,
    depth=6,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=False
)

cat_model.fit(X_train, y_train)
y_val_proba_cat = cat_model.predict_proba(X_val)[:, 1]
roc_auc_cat = roc_auc_score(y_val, y_val_proba_cat)
print(f" AUC-ROC CatBoost : {roc_auc_cat:.4f}")

#  Test avec un **Stacking** simple (moyenne XGB + CatBoost)
y_val_proba_ensemble = (y_val_proba + y_val_proba_cat) / 2
roc_auc_ensemble = roc_auc_score(y_val, y_val_proba_ensemble)
print(f" AUC-ROC Stacking XGB + CatBoost : {roc_auc_ensemble:.4f}")

#  Prédictions sur le test set
test_proba_xgb = xgb_model.predict_proba(test_df)[:, 1]
test_proba_cat = cat_model.predict_proba(test_df)[:, 1]

#  **Stacking final** : Moyenne des deux modèles
test_proba_final = (test_proba_xgb + test_proba_cat) / 2

#  Génération du fichier de soumission
submission_df = pd.DataFrame({'id': test_ids, 'smoking': test_proba_final})
submission_df.to_csv("sample_submission.csv", index=False)

print(" Fichier de soumission prêt !")
