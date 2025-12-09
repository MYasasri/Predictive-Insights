# Predictive-Insights
# Drug Treatment Outcome Classification

## 📌 Project Overview

This project applies **classification algorithms** to predict whether a patient’s treatment leads to **good improvement** based on their demographic details, medical condition, and prescribed drug information.  
The work is done as part of the **Predictive Analytics (INT234)** course.

The model predicts a binary target:

- `Good_Improvement = 1` → patient’s improvement score is **greater than or equal to the median**
- `Good_Improvement = 0` → patient’s improvement score is **below the median**

---

## 🧾 Dataset Description

The dataset used is: `real_drug_dataset.csv` :contentReference[oaicite:0]{index=0}

Each row represents one patient and their treatment details.

**Columns:**

- `Patient_ID` – Unique identifier for each patient (not used for prediction)
- `Age` – Age of the patient (integer)
- `Gender` – Patient gender (categorical)
- `Condition` – Medical condition being treated (categorical)
- `Drug_Name` – Name of the prescribed drug (categorical)
- `Dosage_mg` – Drug dosage in milligrams (numeric)
- `Treatment_Duration_days` – Duration of treatment in days (numeric)
- `Side_Effects` – Observed side effects (categorical)
- `Improvement_Score` – Numeric score representing treatment improvement (float)
- `Good_Improvement` – **Derived binary label**:  
  - `1` if `Improvement_Score >= median`  
  - `0` otherwise

---

## 🎯 Problem Statement

Given patient and treatment information (`Age`, `Gender`, `Condition`, `Drug_Name`, `Dosage_mg`, `Treatment_Duration_days`),  
**can we classify whether the patient will have good improvement (`Good_Improvement`) after treatment?**

---

## 🛠️ Methodology

### 1. Data Loading

- The dataset is loaded using **pandas** from `real_drug_dataset.csv`.

### 2. Exploratory Data Analysis (EDA)

- `head()`, `info()`, and `describe()` are used to understand:
  - Data types
  - Missing values
  - Basic statistics (mean, std, min, max)

### 3. Data Preprocessing

1. **Normalization (Min-Max Scaling)**  
   Applied to continuous numeric features:
   - `Treatment_Duration_days`
   - `Dosage_mg`
   - `Improvement_Score`

   Using:
   ```python
   from sklearn.preprocessing import MinMaxScaler
2. **Label Encoding (Categorical to Numeric)**

The following categorical columns are converted into numeric values using **LabelEncoder**:

- Gender  
- Condition  
- Drug_Name  
- Side_Effects  

Using:

python
from sklearn.preprocessing import LabelEncoder

🧪 Feature Selection
Features (X):
Age
Dosage_mg
Treatment_Duration_days
Gender
Condition
Drug_Name
Label (y):
Good_Improvement

✂️ Train–Test Split

80% training data
20% testing data
random_state = 42 for reproducibility

🤖 Models Implemented
1️⃣ K-Nearest Neighbors (KNN)

n_neighbors = 10
Classifies based on distances of nearest neighbors.

2️⃣ Support Vector Machine (SVM)

Kernel: rbf
Finds the best hyperplane to separate the classes.

3️⃣ Naive Bayes (GaussianNB)

Probabilistic classifier using Bayes’ theorem.
Assumes Gaussian distribution for numeric features.

4️⃣ Decision Tree Classifier

Splits data based on feature thresholds.
random_state = 42 used for reproducibility.

📊 Evaluation Metrics

For each model, the following metrics are computed:
Confusion Matrix
Accuracy Score
Classification Report, including:
Precision
Recall
F1-score



📈 Results
Model	Accuracy (%)	Notes
K-Nearest Neighbors (KNN)	55.50%	
Support Vector Machine (SVM)	48.00%	
Naive Bayes (GaussianNB)	51.00%	
Decision Tree Classifier	53.50%	

You may also include confusion matrices and classification reports below this section.

💻 Tech Stack

Language: Python
Libraries Used:
numpy
pandas
matplotlib
seaborn
scikit-learn (sklearn)


👩‍💻 Author

Name: M Yasasri
Course: INT234 – Predictive Analytics
Institution: Lovely Professional University
