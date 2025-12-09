import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df=pd.read_csv(r"E:\LOVELY PROFFESTIONAL UNIVERSITY\SEM 5\INT234 PREDICTIVE ANALYTICS\Project\real_drug_dataset.csv")
pd.set_option('future.no_silent_downcasting', True)

'''EDA'''
print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#Normalisation
print(df.head())
df.info()
print("Before Normalisation:- \n",df.describe())

scaler = MinMaxScaler()
df[['Treatment_Duration_days','Dosage_mg','Improvement_Score']] = scaler.fit_transform(
    df[['Treatment_Duration_days','Dosage_mg','Improvement_Score']]
)

print("After Normalisation:- \n",df.describe())

#Label Encoding
le = LabelEncoder()
cat_cols = df[['Gender','Condition','Drug_Name','Side_Effects']]
for col in cat_cols:
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")


x = df[['Age', 'Dosage_mg', 'Treatment_Duration_days','Gender', 'Condition', 'Drug_Name'
        ]]
median_score = df['Improvement_Score'].median()
df['Good_Improvement'] = (df['Improvement_Score'] >= median_score).astype(int)

y= df['Good_Improvement']


np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

'''KNN''' 
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train, y_train)
pr1 = knn.predict(x_test)
print('Confusion Matrix (KNN):-\n', confusion_matrix(y_test, pr1))
print("Accuracy (KNN): ", accuracy_score(y_test, pr1)*100)
print("\nClassification Report (KNN):\n", classification_report(y_test, pr1, zero_division=0))

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

'''SVM''' 
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(x_train, y_train)
pr2 = svm_model.predict(x_test)
print("\nConfusion Matrix (SVM):\n", confusion_matrix(y_test, pr2))
print("Accuracy (SVM):", accuracy_score(y_test, pr2)*100)
print("\nClassification Report (SVM):\n", classification_report(y_test, pr2, zero_division=0))

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

'''Naive bayes''' 
nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
pr3= nb_model.predict(x_test)
print("\nConfusion Matrix (Naive Bayes):\n", confusion_matrix(y_test, pr3))
print("Accuracy (Naive Bayes): ", accuracy_score(y_test, pr3)*100)
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, pr3, zero_division=0))

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

'''Decision tree''' 
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)
pr4 = dt.predict(x_test)
print("\nConfusion Matrix (Decision tree):\n", confusion_matrix(y_test, pr4))
print("Accuracy (Decision tree):", accuracy_score(y_test, pr4)*100)
print("\nClassification Report (Decision tree):\n", classification_report(y_test, pr4, zero_division=0))
