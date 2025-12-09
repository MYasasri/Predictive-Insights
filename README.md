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
2. Label Encoding (Categorical to Numeric)
Applied to:

Gender

Condition

Drug_Name

Side_Effects

Using:

from sklearn.preprocessing import LabelEncoder


Feature Selection

Features (X):

Age

Dosage_mg

Treatment_Duration_days

Gender

Condition

Drug_Name

Label (y):

Good_Improvement

Train–Test Split

80% training data

20% testing data

random_state=42 for reproducibility

🤖 Models Implemented

The following classification algorithms are used:

K-Nearest Neighbors (KNN)

n_neighbors = 10

Distance-based classification based on nearest neighbors.

Support Vector Machine (SVM)

Kernel: rbf

Finds an optimal hyperplane to separate the two classes.

Naive Bayes (GaussianNB)

Probabilistic classifier based on Bayes’ theorem.

Works well for continuous features assuming Gaussian distribution.

Decision Tree Classifier

Tree-based model that splits data using feature thresholds.

random_state=42 for reproducibility.

📊 Evaluation Metrics

For each model, the following metrics are calculated:

Confusion Matrix

Accuracy Score

Classification Report:

Precision

Recall

F1-score

Support

These are computed using:

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

📈 Results

🔧 Note: Fill in the actual numbers after you run the code.

Model	Accuracy (%)	Notes
K-Nearest Neighbors (KNN)	55.50000000000001
Support Vector Machine (SVM)	48.0	
Naive Bayes (GaussianNB)	51.0	
Decision Tree Classifier	53.5	

You can also copy-paste the confusion matrices and classification reports into this section if needed.

💻 Tech Stack

Language: Python

Libraries:

numpy

pandas

matplotlib

seaborn

scikit-learn (sklearn)

▶️ How to Run the Project

Clone the repository / download the files

Make sure you have:

CA2.py (main code file)

real_drug_dataset.csv (dataset) in the same folder or update the file path in the code.

Install required libraries

pip install numpy pandas matplotlib seaborn scikit-learn


Run the script

python CA2.py


Check the terminal output for:

EDA summaries

Normalization before/after

Confusion matrices

Accuracy and classification reports for each model

📌 Future Improvements

Try hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for KNN, SVM, and Decision Tree.

Compare additional models like:

Random Forest

Gradient Boosting

Add visualizations:

Feature importance (for Decision Tree)

ROC curves

Build a simple GUI or web app to input patient details and get prediction.

👩‍💻 Author

Name: M Yasasri

Course: INT234 – Predictive Analytics

Institution: Lovely Professional University
