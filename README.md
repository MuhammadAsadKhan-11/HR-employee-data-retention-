# Employee Attrition Prediction using Logistic Regression

A machine learning project that predicts employee attrition (turnover) using logistic regression trained on HR data. The model identifies which employees are likely to leave the company based on factors like satisfaction level, workload, salary, and tenure.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Pipeline](#model-pipeline)
- [Results](#results)
- [Saving the Model](#saving-the-model)
- [Dependencies](#dependencies)

---

## Overview

Employee attrition is a costly challenge for organizations. This project uses **Logistic Regression** to build a binary classifier that predicts whether an employee will leave (`left = 1`) or stay (`left = 0`). The workflow includes:

- Data loading and preprocessing
- Exploratory data analysis with visualizations
- Feature engineering and encoding
- Model training using a Scikit-learn Pipeline (StandardScaler + LogisticRegression)
- Per-feature probability plots
- Model serialization with Joblib

---

## Dataset

**File:** `HR_comma_sep.csv`

This is the well-known [HR Analytics dataset](https://www.kaggle.com/datasets/ludobenistant/hr-analytics), commonly used for attrition modeling.

### Features

| Column | Description |
|---|---|
| `satisfaction_level` | Employee satisfaction score (0.0 – 1.0) |
| `last_evaluation` | Score from the last performance review (0.0 – 1.0) |
| `number_project` | Number of projects the employee worked on |
| `average_montly_hours` | Average monthly hours worked |
| `time_spend_company` | Number of years spent at the company |
| `Work_accident` | Whether the employee had a workplace accident (0/1) |
| `promotion_last_5years` | Whether the employee was promoted in the last 5 years (0/1) |
| `Department` | Department name (categorical) |
| `salary` | Salary level — Low, Medium, or High (categorical → encoded) |
| `left` | **Target variable** — whether the employee left (1) or stayed (0) |

---

## Project Structure

```
├── hr_logisticRegression.ipynb       # Main Jupyter Notebook
├── HR_comma_sep.csv                  # HR dataset (required)
├── logistic_model_joblib             # Serialized trained model (generated)
├── impact of salary on Employee      # Salary vs. attrition chart (generated)
├── Employee Retention By department  # Department vs. attrition chart (generated)
├── Feature vs Predicted Probability of leaving  # Feature probability plots (generated)
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies

It is recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### 3. Add the dataset

Place `HR_comma_sep.csv` in the root of the project directory before running the notebook.

---

## Usage

Launch Jupyter Notebook and open the main file:

```bash
jupyter notebook hr_logisticRegression.ipynb
```

Run all cells in order. The notebook will:

1. Load and explore the data
2. Encode categorical variables
3. Visualize attrition patterns
4. Train a logistic regression model
5. Generate prediction probability plots
6. Save the trained model to disk

### Making a single prediction

After training, you can predict for a new employee directly:

```python
# Input order: [satisfaction_level, last_evaluation, number_project,
#               average_montly_hours, time_spend_company,
#               Work_accident, promotion_last_5years, salary]

model.predict([[0.80, 0.86, 5, 262, 6, 0, 0, 2]])
# Output: array([0])  →  Employee is predicted to stay
```

---

## Exploratory Data Analysis

Two bar charts are generated to visualize attrition patterns before modeling:

### 1. Impact of Salary on Employee Retention

Groups employees by salary tier (Low / Medium / High) and plots the percentage who stayed vs. left. This highlights whether compensation is a significant driver of attrition.

### 2. Employee Retention by Department

Groups employees by department and sorts by attrition rate (descending). This reveals which teams have the highest turnover risk.

Both charts are saved as image files in the project directory.

---

## Model Pipeline

The model uses a Scikit-learn `Pipeline` to chain preprocessing and classification:

```
Raw Features
    │
    ▼
StandardScaler        ← Normalizes all numeric features to zero mean, unit variance
    │
    ▼
LogisticRegression    ← Binary classifier (max_iter=1000)
    │
    ▼
Prediction (0 = Stay, 1 = Leave)
```

### Train/Test Split

| Parameter | Value |
|---|---|
| Test size | 20% |
| Random state | 40 |

### Features used for training

The `Department` column (multi-class categorical) and the target `left` are dropped before training. The `salary` column is label-encoded (Low → 0, Medium → 1 or 2, High → 2 or 1) prior to model input.

---

## Results

After training, predicted probabilities are plotted against each feature for the test set. Eight subplots are generated (one per feature), showing:

- **Blue dots** — actual outcomes (0 or 1) for test samples
- **Orange curve** — model's predicted probability of leaving, sorted along the feature axis

These plots help interpret how each feature influences the model's confidence that an employee will leave.

The final figure is saved as `"Feature vs Predicted Probability of leaving"`.

---

## Saving the Model

The trained pipeline is serialized using **Joblib** for later reuse:

```python
import joblib
joblib.dump(model, "logistic_model_joblib")
```

To reload and use the model in another script:

```python
import joblib
model = joblib.load("logistic_model_joblib")
prediction = model.predict([[0.5, 0.7, 4, 200, 3, 0, 0, 1]])
```

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Data visualization |
| `scikit-learn` | Preprocessing, model training, and pipeline |
| `joblib` | Model serialization |

---

## Potential Improvements

- Handle the `Department` column via **One-Hot Encoding** rather than dropping it, to capture department-level signal
- Evaluate model performance with metrics such as accuracy, precision, recall, F1-score, and AUC-ROC
- Compare against other classifiers (Random Forest, XGBoost, SVM)
- Add cross-validation for more robust performance estimates
- Build an interactive prediction interface (e.g., Streamlit or Gradio)

---

## License

This project is open source. Add your preferred license here (e.g., MIT).
