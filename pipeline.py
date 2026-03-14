"""
Car Purchase Classification – Preprocessing & Model Pipeline

Loads Data/car_data.csv, preprocesses (drop User ID, encode Gender, scale Age/AnnualSalary),
splits with stratification, runs 5-fold CV, fits Logistic Regression, Random Forest, and Gradient Boosting,
evaluates on test set, selects best by ROC-AUC, and saves best pipeline with joblib.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

# --- 1. Load and prepare ---
df = pd.read_csv("Car-Purchase-Classification\Data\car_data.csv")
df = df.drop(columns=["User ID"])
X = df.drop(columns=["Purchased"])
y = df["Purchased"]

# --- 2. Preprocessor (template; each pipeline gets its own clone) ---
numeric_features = ["Age", "AnnualSalary"]
categorical_features = ["Gender"]

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )

# --- 3. Train/test split (stratified) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Build pipelines (preprocessor + model), each with its own preprocessor ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

pipelines = {
    "Logistic Regression": Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    ),
    "Random Forest": Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    ),
    "Gradient Boosting": Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("model", GradientBoostingClassifier(random_state=42)),
        ]
    ),
}

# --- 5. Cross-validation on training set ---
cv_results_list = []
for name, pipeline in pipelines.items():
    scores = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
    )
    row = {"Model": name}
    for s in scoring:
        key = "test_" + s
        if key in scores:
            row[f"{s}_mean"] = np.mean(scores[key])
            row[f"{s}_std"] = np.std(scores[key])
    cv_results_list.append(row)

print("Car Purchase Classification – 5-fold CV (train set)\n")
cv_df = pd.DataFrame(cv_results_list)
# Flatten for display: Model, then each metric as mean ± std
cv_display = pd.DataFrame({"Model": cv_df["Model"]})
for s in scoring:
    cv_display[f"{s}"] = cv_df[f"{s}_mean"].apply(lambda x: f"{x:.4f}") + " ± " + cv_df[f"{s}_std"].apply(lambda x: f"{x:.4f}")
print(cv_display.to_string(index=False))

# --- 6. Fit on full train, evaluate on test ---
results = []
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0
    )
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results.append(
        {
            "Model": name,
            "Pipeline": pipeline,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC-AUC": auc,
            "Confusion Matrix": cm,
        }
    )

print("\nCar Purchase Classification – Model comparison (test set)\n")
comparison = pd.DataFrame(
    [
        {
            "Model": r["Model"],
            "Accuracy": f"{r['Accuracy']:.4f}",
            "Precision": f"{r['Precision']:.4f}",
            "Recall": f"{r['Recall']:.4f}",
            "F1": f"{r['F1']:.4f}",
            "ROC-AUC": f"{r['ROC-AUC']:.4f}",
        }
        for r in results
    ]
)
print(comparison.to_string(index=False))
print("\nConfusion matrices (TN, FP / FN, TP):")
for r in results:
    print(f"  {r['Model']}: {r['Confusion Matrix'].tolist()}")

# --- 7. Select best by test ROC-AUC and save ---
best = max(results, key=lambda r: r["ROC-AUC"])
best_pipeline = best["Pipeline"]
best_name = best["Model"]
model_path = "model.joblib"
joblib.dump(best_pipeline, model_path)
print(f"\nSaved best model ({best_name}, test ROC-AUC={best['ROC-AUC']:.4f}) to {model_path}")
