import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression

DATA_PATH = "data/diabetic_data.csv"
MODEL_PATH = "models/readmission_model.joblib"

def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset introuvable: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    df = df.replace("?", np.nan)


    if "readmitted" not in df.columns:
        raise ValueError("Colonne 'readmitted' introuvable dans le CSV.")

    df["target_readmit_30"] = (df["readmitted"] == "<30").astype(int)

    df = df.drop(columns=["encounter_id", "patient_nbr"], errors="ignore")

    y = df["target_readmit_30"]
    X = df.drop(columns=["target_readmit_30", "readmitted"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print("ROC-AUC:", round(auc, 4))
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Modèle sauvegardé -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
