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
from sklearn.base import clone

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------------
# 1) Chargement des données
# -------------------------------------------------------------------
train_df = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/train.csv")
test_df  = pd.read_csv("/kaggle/input/competition-epsi-2025-ds-ml-g-3-g-4/test.csv")

y = train_df["smoking"].values
train_ids = train_df["id"].values
test_ids  = test_df["id"].values

train_df.drop(["id","smoking"], axis=1, inplace=True)
test_df.drop(["id"], axis=1, inplace=True)

# -------------------------------------------------------------------
# 2) Normalisation (optionnelle)
# -------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(train_df)
X_test = scaler.transform(test_df)

# -------------------------------------------------------------------
# 3) Fonction OOF (k-fold) identique
# -------------------------------------------------------------------
def get_oof_and_test_preds(model, X, y, X_test, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        this_model = clone(model)
        
        # LightGBM
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
        
        oof_preds[val_idx] = this_model.predict_proba(X_val)[:,1]
        test_preds += this_model.predict_proba(X_test)[:,1] / n_splits
        
    return oof_preds, test_preds

# -------------------------------------------------------------------
# 4) Instanciation des 3 modèles de base
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
    "verbosity": 0
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
# 5) K-Fold : OOF + Test pour XGB, LGB, Cat
# -------------------------------------------------------------------
oof_xgb, test_xgb = get_oof_and_test_preds(xgb_model, X, y, X_test)
oof_lgb, test_lgb = get_oof_and_test_preds(lgb_model, X, y, X_test)
oof_cat, test_cat = get_oof_and_test_preds(cat_model, X, y, X_test)

print(f"OOF AUC XGBoost : {roc_auc_score(y, oof_xgb):.4f}")
print(f"OOF AUC LightGBM: {roc_auc_score(y, oof_lgb):.4f}")
print(f"OOF AUC CatBoost: {roc_auc_score(y, oof_cat):.4f}")

# -------------------------------------------------------------------
# 6) Méta-modèle : XGBoost (au lieu d'une régression logistique)
# -------------------------------------------------------------------
meta_train = np.column_stack([oof_xgb, oof_lgb, oof_cat])
meta_test  = np.column_stack([test_xgb, test_lgb, test_cat])

meta_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "use_label_encoder": False,
    "n_estimators": 200,       # tu peux ajuster
    "learning_rate": 0.05,
    "max_depth": 3,
    "random_state": 42,
    "verbosity": 0
}
meta_model = xgb.XGBClassifier(**meta_params)

meta_model.fit(
    meta_train, y,
    eval_set=[(meta_train, y)],
    early_stopping_rounds=50,
    verbose=False
)

oof_meta = meta_model.predict_proba(meta_train)[:, 1]
auc_meta = roc_auc_score(y, oof_meta)
print(f"OOF AUC Méta-modèle (XGB) : {auc_meta:.4f}")

# -------------------------------------------------------------------
# 7) Courbe ROC finale
# -------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y, oof_meta)
plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"Méta XGB (AUC={auc_meta:.4f})", color="blue")
plt.plot([0,1],[0,1], '--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacking avec méta XGB")
plt.legend(loc="lower right")
plt.show()

# -------------------------------------------------------------------
# 8) Prédictions finales + fichier
# -------------------------------------------------------------------
test_preds_final = meta_model.predict_proba(meta_test)[:, 1]

submission_df = pd.DataFrame({
    "id": test_ids,
    "smoking": test_preds_final
})
submission_df.to_csv("sample_submission.csv", index=False)
print("Fichier de soumission prêt : sample_submission.csv")

#8987