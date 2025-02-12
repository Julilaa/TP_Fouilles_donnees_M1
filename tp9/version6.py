import numpy as np
import pandas as pd

# Modèles
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------
# 1) Chargement des données (on ne change pas tes noms de fichiers)
# -------------------------------------------------------------------
train_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/train.csv")
test_df  = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/test.csv")

# On sépare la target et l'id
y = train_df["smoking"].values
train_ids = train_df["id"].values
test_ids  = test_df["id"].values

train_df.drop(["id", "smoking"], axis=1, inplace=True)
test_df.drop(["id"], axis=1, inplace=True)

# -------------------------------------------------------------------
# 2) Préprocessing : Normalisation (optionnelle)
# -------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(train_df)
X_test = scaler.transform(test_df)

# -------------------------------------------------------------------
# 3) Fonction k-fold pour obtenir OOF et prédictions test moyennées
# -------------------------------------------------------------------
def get_oof_and_test_preds(model, X, y, X_test, n_splits=5, random_state=42):
    """
    Entraîne 'model' en k-fold stratifié. Retourne :
      - oof_preds : array de taille len(X), prédictions OOF pour chaque exemple train
      - test_preds : moyenne des prédictions sur X_test
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # On clone pour chaque fold, afin de repartir d'un modèle “neuf”
        this_model = clone(model)
        
        # LightGBM : on gère les callbacks + verbose
        if isinstance(this_model, lgb.LGBMClassifier):
            this_model.set_params(verbose=-1)
            this_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
        # CatBoost
        elif isinstance(this_model, cb.CatBoostClassifier):
            this_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                early_stopping_rounds=50,
                verbose=False
            )
        # XGBoost ou autre
        else:
            this_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        
        # On stocke les prédictions OOF
        oof_preds[val_idx] = this_model.predict_proba(X_val)[:, 1]
        
        # On additionne les prédictions test (on divisera après)
        test_preds += this_model.predict_proba(X_test)[:, 1] / n_splits
    
    return oof_preds, test_preds

# -------------------------------------------------------------------
# 4) Instanciation des 3 modèles de base (avec tes hyperparamètres)
# -------------------------------------------------------------------
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 4,
    "colsample_bytree": 0.67,
    "subsample": 0.68,
    "gamma": 3.1,
    "reg_lambda": 0.63,
    "random_state": 42,
    "verbosity": 0  # pour couper logs XGBoost
}
xgb_model = xgb.XGBClassifier(**xgb_params)

lgb_model = lgb.LGBMClassifier(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    colsample_bytree=0.7,
    subsample=0.7,
    random_state=42
)

cat_model = cb.CatBoostClassifier(
    iterations=600,
    learning_rate=0.05,
    depth=6,
    eval_metric="AUC",
    random_seed=42,
    verbose=False
)

# -------------------------------------------------------------------
# 5) K-Fold : on récupère OOF + Test pour XGB, LGB, Cat
# -------------------------------------------------------------------
oof_xgb, test_xgb = get_oof_and_test_preds(xgb_model, X, y, X_test)
oof_lgb, test_lgb = get_oof_and_test_preds(lgb_model, X, y, X_test)
oof_cat, test_cat = get_oof_and_test_preds(cat_model, X, y, X_test)

auc_xgb = roc_auc_score(y, oof_xgb)
auc_lgb = roc_auc_score(y, oof_lgb)
auc_cat = roc_auc_score(y, oof_cat)

print(f"OOF AUC XGBoost : {auc_xgb:.4f}")
print(f"OOF AUC LightGBM: {auc_lgb:.4f}")
print(f"OOF AUC CatBoost: {auc_cat:.4f}")

# -------------------------------------------------------------------
# 6) On entraîne un méta-modèle simple : LogisticRegression
#    Sur les OOF
# -------------------------------------------------------------------
meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
meta_test  = np.column_stack([test_xgb, test_lgb, test_cat])

meta_model = LogisticRegression()
meta_model.fit(meta_train, y)

# OOF final du stacking (c'est "in-sample" pour le méta)
oof_meta = meta_model.predict_proba(meta_train)[:, 1]
auc_meta = roc_auc_score(y, oof_meta)
print(f"OOF AUC Méta-modèle : {auc_meta:.4f}")

# -------------------------------------------------------------------
# 7) Courbe ROC finale
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y, oof_meta)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"Méta-modèle (AUC={auc_meta:.4f})", color="blue")
plt.plot([0,1],[0,1], '--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacking Simple")
plt.legend(loc="lower right")
plt.show()

# -------------------------------------------------------------------
# 8) Prédictions finales sur le test + fichier submission
# -------------------------------------------------------------------
test_preds_final = meta_model.predict_proba(meta_test)[:, 1]

submission_df = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds_final
})
submission_df.to_csv("sample_submission.csv", index=False)

print("Fichier de soumission prêt : sample_submission.csv")
