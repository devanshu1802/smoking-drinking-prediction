<p align="center">
  <h1 align="center">🫁 Smoking & Drinking Prediction Using Body Signals</h1>
  <p align="center">
    <em>A machine learning pipeline to predict smoking and drinking habits from measurable health indicators — because patients don't always tell the truth.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10"/>
    <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
    <img src="https://img.shields.io/badge/XGBoost-Latest-006600?style=for-the-badge" alt="XGBoost"/>
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker"/>
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  </p>
</p>

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Key Results](#-key-results)
- [Dataset Overview](#-dataset-overview)
- [Project Architecture](#-project-architecture)
- [Models & Methodology](#-models--methodology)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Results & Evaluation](#-results--evaluation)
- [Tech Stack](#-tech-stack)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Author](#-author)

---

## 🎯 Problem Statement

When patients are diagnosed with respiratory or health conditions, they are often **reluctant to disclose their smoking and drinking habits** — due to social stigma, fear of judgment, or other personal reasons. This leads to misinformation and makes it significantly harder for medical practitioners to provide accurate treatment.

**This project addresses that gap** by leveraging measurable body signals — such as blood pressure, cholesterol levels, hemoglobin, liver enzymes, and other clinical indicators — to **predict a patient's smoking and drinking status** without relying on self-reported data.

> **Goal:** Build a robust, multi-model classification pipeline that accurately predicts smoking status and drinking behavior based on physiological and demographic features.

---

## 🏆 Key Results

| Model | Training Accuracy | Testing Accuracy |
|:------|:-----------------:|:----------------:|
| Logistic Regression | 72.61% | 72.78% |
| Random Forest | 77.62% | 73.13% |
| Support Vector Machine (SVC) | 74.53% | 73.34% |
| XGBoost | 75.06% | **73.75%** |
| Gradient Boosting | 74.89% | 73.44% |
| **Stacking Ensemble** | **76.30%** | **73.79%** |

> ✅ **Best Performer:** The **Stacking Ensemble** classifier achieved the highest testing accuracy at **73.79%**, combining the strengths of multiple base learners for more stable and generalized predictions.

---

## 📊 Dataset Overview

- **Source:** Health screening dataset containing body signal measurements
- **Size:** 49,999 records × 24 features
- **Target Variables:**
  - `SMK_stat_type_cd` — Smoking status (1: Never, 2: Used to, 3: Currently smoking)
  - `DRK_YN` — Drinking status (Y/N)
- **Missing Values:** None ✅
- **Duplicate Records:** None ✅

### Feature Breakdown

| Category | Features |
|:---------|:---------|
| **Demographics** | `sex`, `age` |
| **Anthropometrics** | `height`, `weight`, `waistline` |
| **Sensory** | `sight_left`, `sight_right`, `hear_left`, `hear_right` |
| **Cardiovascular** | `SBP` (Systolic BP), `DBP` (Diastolic BP) |
| **Blood Chemistry** | `BLDS` (Blood Sugar), `tot_chole`, `HDL_chole`, `LDL_chole`, `triglyceride` |
| **Hematology** | `hemoglobin` |
| **Renal** | `urine_protein`, `serum_creatinine` |
| **Liver Function** | `SGOT_AST`, `SGOT_ALT`, `gamma_GTP` |

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      DATA PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │   Data   │──▶│  Data    │──▶│ Feature  │──▶│ Feature │ │
│  │ Loading  │   │ Cleaning │   │Engineerin│   │ Scaling │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
│                                                      │      │
│  ┌───────────────────────────────────────────────────▼──┐   │
│  │               MODEL TRAINING & TUNING                │   │
│  │                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │   │
│  │  │  Logistic   │  │   Random    │  │     SVM     │  │   │
│  │  │ Regression  │  │   Forest    │  │   (SVC)     │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │   │
│  │         │                │                │         │   │
│  │  ┌──────┴──────┐  ┌──────┴──────┐         │         │   │
│  │  │  XGBoost    │  │  Gradient   │         │         │   │
│  │  │ Classifier  │  │  Boosting   │         │         │   │
│  │  └──────┬──────┘  └──────┬──────┘         │         │   │
│  │         │                │                │         │   │
│  │         ▼                ▼                ▼         │   │
│  │  ┌──────────────────────────────────────────────┐   │   │
│  │  │         STACKING ENSEMBLE CLASSIFIER         │   │   │
│  │  │    (Final Meta-Learner: LogisticRegression)  │   │   │
│  │  └──────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              EVALUATION & REPORTING                  │   │
│  │  • Accuracy Score  • Classification Report           │   │
│  │  • Cross-Validation  • RandomizedSearchCV            │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Models & Methodology

### Data Preprocessing
- **Column Renaming:** Raw column names were mapped to descriptive, human-readable names for clarity
- **Label Encoding:** Categorical variables (`sex`, `DRK_YN`) converted to numerical representations
- **Feature Scaling:** `StandardScaler` applied for normalization across all numeric features
- **Train-Test Split:** 80/20 stratified split for robust evaluation

### Hyperparameter Tuning
All models were fine-tuned using **`RandomizedSearchCV`** with:
- **5-Fold Cross-Validation** for stable performance estimates
- **Scoring Metric:** Accuracy
- **Random State:** 42 (for reproducibility)

### Models Implemented

| # | Model | Tuning Strategy | Key Hyperparameters |
|:-:|:------|:----------------|:--------------------|
| 1 | **Logistic Regression** | RandomizedSearchCV | `C`, `penalty`, `solver`, `max_iter` |
| 2 | **Random Forest Classifier** | RandomizedSearchCV | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| 3 | **SVM (SVC)** | RandomizedSearchCV | `C`, `kernel`, `gamma` |
| 4 | **XGBoost Classifier** | RandomizedSearchCV | `n_estimators`, `learning_rate`, `max_depth`, `subsample` |
| 5 | **Gradient Boosting Classifier** | RandomizedSearchCV | `n_estimators`, `learning_rate`, `max_depth`, `min_samples_leaf` |
| 6 | **Stacking Classifier** | Ensemble of above | Meta-learner: Logistic Regression |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- pip or conda
- Docker (optional, for containerized execution)

### Option 1: Local Setup

```bash
# Clone the repository
git clone https://github.com/devanshu1802/smoking-drinking-prediction.git
cd smoking-drinking-prediction

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook Smoking_Drinking_Prediction.ipynb
```

### Option 2: Docker

```bash
# Build the Docker image
docker build -t smoking-drinking-prediction .

# Run the container
docker run -p 8888:8888 smoking-drinking-prediction
```

Then open `http://localhost:8888` in your browser to access the notebook.

---

## 📁 Project Structure

```
smoking-drinking-prediction/
│
├── Smoking_Drinking_Prediction.ipynb   # Main analysis notebook (end-to-end pipeline)
├── smoking_drinking_dataset.csv        # Dataset (49,999 records × 24 features)
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Docker containerization config
├── README.md                           # Project documentation (you are here)
└── .gitignore                          # Git ignore rules
```

---

## 📈 Exploratory Data Analysis

The notebook includes comprehensive EDA covering:

- **Distribution Analysis** — Histograms, density plots, and box plots across all features
- **Correlation Analysis** — Heatmap visualization of feature-to-feature correlations
- **Target Variable Analysis** — Class distribution for both smoking and drinking status
- **Feature Relationships** — Pairwise and grouped visualizations to uncover key patterns
- **Outlier Detection** — Identification and handling of extreme values in clinical measurements

Key insights:
- Gender shows strong correlation with both smoking and drinking behavior
- Liver enzymes (`gamma_GTP`, `SGOT_ALT`) are strong predictors of drinking status
- Hemoglobin levels differ significantly between smokers and non-smokers
- Blood pressure metrics contribute moderately to prediction accuracy

---

## 📊 Results & Evaluation

Each model was evaluated with:
- **Training & Testing Accuracy** — To check for overfitting
- **Classification Report** — Precision, Recall, F1-Score per class
- **Cross-Validation Scores** — 5-fold CV for generalization assessment

### Final Model Comparison

```
Model                    | Train Acc. | Test Acc. | Overfitting Gap
─────────────────────────┼────────────┼───────────┼────────────────
Logistic Regression      │   72.61%   │  72.78%   │     -0.17%
Random Forest            │   77.62%   │  73.13%   │      4.49%
SVM (SVC)                │   74.53%   │  73.34%   │      1.19%
XGBoost                  │   75.06%   │  73.75%   │      1.31%
Gradient Boosting        │   74.89%   │  73.44%   │      1.45%
Stacking Ensemble        │   76.30%   │  73.79%   │      2.51%
```

> **Key Takeaway:** The Stacking Ensemble achieves the best test performance while maintaining a reasonable overfitting gap, making it the most reliable model for deployment.

---

## 🛠️ Tech Stack

| Category | Technologies |
|:---------|:-------------|
| **Language** | Python 3.10 |
| **Data Manipulation** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, XGBoost |
| **ML Utilities** | `StandardScaler`, `LabelEncoder`, `train_test_split`, `RandomizedSearchCV`, `cross_val_score` |
| **Containerization** | Docker |
| **Environment** | Jupyter Notebook |

---

## 🔮 Future Improvements

- [ ] **Deep Learning Integration** — Explore neural network architectures (e.g., TabNet, 1D-CNN) for potentially higher accuracy
- [ ] **Feature Engineering** — Derive new features like BMI, blood pressure categories, and cholesterol ratios
- [ ] **SMOTE / Class Balancing** — Address potential class imbalance using oversampling techniques
- [ ] **Model Explainability** — Integrate SHAP / LIME for interpretable predictions
- [ ] **REST API Deployment** — Build a Flask/FastAPI endpoint for real-time predictions
- [ ] **Full CI/CD Pipeline** — Automate testing, training, and deployment workflows
- [ ] **Larger Dataset** — Leverage the complete dataset (currently using a subset of ~50K records) for improved generalization

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/your-feature`)
3. **Commit** your changes (`git commit -m 'Add: your feature description'`)
4. **Push** to the branch (`git push origin feature/your-feature`)
5. **Open** a Pull Request

---

## 👤 Author

**Devanshu Singh**

- GitHub: [@devanshu1802](https://github.com/devanshu1802)

---

<p align="center">
  <sub>⭐ If you found this project useful, consider giving it a star!</sub>
</p>
