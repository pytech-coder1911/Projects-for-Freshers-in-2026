# ============================================================
# PROJECT 1: Customer Churn Prediction
# Stack: XGBoost, SHAP, MLflow, Streamlit
# Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# ============================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ------ 1. Load & Clean Data ------
df = pd.read_csv("telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)
df.drop("customerID", axis=1, inplace=True)
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# ------ 2. Encode Categoricals ------
cat_cols = df.select_dtypes(include="object").columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ------ 3. Train/Test Split ------
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------ 4. Train Model with MLflow Tracking ------
with mlflow.start_run(run_name="xgboost_churn"):
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba)
    print(classification_report(y_test, preds))
    print(f"AUC-ROC: {auc:.4f}")

    mlflow.log_metric("auc_roc", auc)
    mlflow.log_param("n_estimators", 200)
    mlflow.xgboost.log_model(model, "churn_model")

# ------ 5. SHAP Explainability ------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

model.save_model("churn_model.json")
print("Model saved as churn_model.json")
