"""
Car Purchase Classification – Streamlit UI

Simple UI to test the model: enter Gender, Age, AnnualSalary and get
a prediction (and probability), or upload a CSV for batch predictions.
"""
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "model.joblib"
REQUIRED_COLUMNS = ["Gender", "Age", "AnnualSalary"]


@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model not found: {MODEL_PATH}. Run `python pipeline.py` first to train and save the model.")
        st.stop()


def main():
    st.set_page_config(page_title="Car Purchase Predictor", page_icon="🚗", layout="centered")
    st.title("🚗 Car Purchase Predictor")
    st.caption("Predict whether a customer will purchase a car from Gender, Age, and Annual Salary.")

    pipeline = load_model()

    # Tabs: single prediction vs batch (CSV)
    tab1, tab2 = st.tabs(["Single prediction", "Batch (CSV)"])

    with tab1:
        st.subheader("Enter customer details")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
            age = st.number_input("Age", min_value=18, max_value=100, value=40, step=1, key="age")
        with col2:
            salary = st.number_input("Annual Salary", min_value=0, max_value=500000, value=80000, step=5000, key="salary")

        if st.button("Predict", type="primary", key="btn_single"):
            X = pd.DataFrame([{"Gender": gender, "Age": age, "AnnualSalary": salary}])
            pred = pipeline.predict(X)[0]
            prob = pipeline.predict_proba(X)[0, 1]
            if pred == 1:
                st.success(f"**Prediction: Will purchase** (probability: {prob:.2%})")
            else:
                st.info(f"**Prediction: Will not purchase** (probability of purchase: {prob:.2%})")

    with tab2:
        st.subheader("Upload CSV for batch prediction")
        st.caption("CSV must have columns: Gender, Age, AnnualSalary")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"], key="upload")
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}. Required: {REQUIRED_COLUMNS}")
            else:
                X = df[REQUIRED_COLUMNS].copy()
                X["Age"] = pd.to_numeric(X["Age"], errors="coerce")
                X["AnnualSalary"] = pd.to_numeric(X["AnnualSalary"], errors="coerce")
                if X.isnull().any().any():
                    st.error("Age and AnnualSalary must be numeric; found missing or invalid values.")
                else:
                    pred = pipeline.predict(X)
                    prob = pipeline.predict_proba(X)[:, 1]
                    out = df.copy()
                    out["Predicted_Purchased"] = pred
                    out["Predicted_Prob"] = prob
                    st.dataframe(out, use_container_width=True)
                    st.download_button(
                        "Download predictions (CSV)",
                        out.to_csv(index=False).encode("utf-8"),
                        file_name="predictions.csv",
                        mime="text/csv",
                        key="download",
                    )


if __name__ == "__main__":
    main()
