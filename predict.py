"""
Car Purchase Classification – Predict script

Loads the saved pipeline (model.joblib), reads new data from a CSV with columns
Gender, Age, AnnualSalary, and writes predictions to an output CSV with
Predicted_Purchased (0/1) and optionally Predicted_Prob.
"""
import argparse
import sys
import pandas as pd
import joblib

REQUIRED_COLUMNS = ["Gender", "Age", "AnnualSalary"]
MODEL_PATH = "model.joblib"


def main():
    parser = argparse.ArgumentParser(description="Predict car purchase for new data.")
    parser.add_argument(
        "--input",
        "-i",
        default="Data/new_customers.csv",
        help="Input CSV with columns Gender, Age, AnnualSalary",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="predictions.csv",
        help="Output CSV with predictions",
    )
    parser.add_argument(
        "--prob",
        action="store_true",
        help="Include probability of class 1 in output",
    )
    args = parser.parse_args()

    # Load pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Run pipeline.py first to train and save the model.", file=sys.stderr)
        sys.exit(1)

    # Read input
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: Input CSV must have columns {REQUIRED_COLUMNS}. Missing: {missing}", file=sys.stderr)
        sys.exit(1)

    X = df[REQUIRED_COLUMNS].copy()
    # Ensure numeric types
    for col in ["Age", "AnnualSalary"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X.isnull().any().any():
        print("Error: Age and AnnualSalary must be numeric; found missing or invalid values.", file=sys.stderr)
        sys.exit(1)

    # Predict
    pred = pipeline.predict(X)
    out = df.copy()
    out["Predicted_Purchased"] = pred
    if args.prob:
        out["Predicted_Prob"] = pipeline.predict_proba(X)[:, 1]

    out.to_csv(args.output, index=False)
    print(f"Predictions written to {args.output} ({len(out)} rows).")


if __name__ == "__main__":
    main()
