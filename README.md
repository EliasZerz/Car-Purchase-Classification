# Car Purchase Classification

Binary classification: predict whether a customer will purchase a car from **Gender**, **Age**, and **Annual Salary**. The project trains several scikit-learn pipelines (logistic regression, random forest, gradient boosting), picks the best by test ROC-AUC, and exposes predictions through a **Streamlit** app and a small **CLI** script.

## Requirements

- Python 3.9+ recommended  
- Dependencies: see [`requirements.txt`](requirements.txt)

```bash
pip install -r requirements.txt
```

## Data

- Training data: `Data/car_data.csv` (target column: `Purchased`).  
- Example batch input: `Data/new_customers.csv`.

## Train the model and save `model.joblib`

`pipeline.py` loads the CSV with a path relative to the **parent** folder of this project (the directory that contains `Car-Purchase-Classification`). Run training from that parent directory (e.g. `car-ml-project`):

```bash
# Windows / macOS / Linux — replace path with your clone location
cd path/to/car-ml-project
python Car-Purchase-Classification/pipeline.py
```

This writes `model.joblib` in the **current working directory** (next to the `Car-Purchase-Classification` folder).

## Run the Streamlit app

**Option A — same layout as training (recommended):** from the parent folder (`car-ml-project`):

```bash
streamlit run Car-Purchase-Classification/app.py
```

**Option B — from inside this repo folder:** ensure `model.joblib` is present here (copy it from the parent folder after training, or train as above and copy), then:

```bash
cd Car-Purchase-Classification
streamlit run app.py
```

The UI supports a single prediction and batch upload; batch CSVs must include columns: `Gender`, `Age`, `AnnualSalary`.

## CLI predictions (optional)

With `model.joblib` on the current working directory (same rules as above):

```bash
python predict.py --input Data/new_customers.csv --output predictions.csv
```

Add `--prob` to include the probability of purchase in the output.

## Project layout

| File / folder   | Role |
|-----------------|------|
| `pipeline.py`   | Load data, CV, test metrics, save best pipeline to `model.joblib` |
| `app.py`        | Streamlit UI |
| `predict.py`    | Batch predictions from CSV |
| `eda.ipynb`     | Exploratory analysis |
| `Data/`         | Datasets |
