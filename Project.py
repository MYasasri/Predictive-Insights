import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv(r"E:\LOVELY PROFFESTIONAL UNIVERSITY\SEM 5\INT234 PREDICTIVE ANALYTICS\Project\student_performance_updated_1000.csv")
pd.set_option('future.no_silent_downcasting', True)

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

print(df.head())
df.info()


print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")
#handling null values
print("\ntotal null values in the dataset before cleaning :-\n",
      df.isnull().sum())
df = df.dropna(subset=['StudentID'])
df = df.dropna(subset=['Name'])

#numerical values
num_cols = ['AttendanceRate', 'StudyHoursPerWeek','PreviousGrade',
            'ExtracurricularActivities','FinalGrade','Study Hours','Attendance (%)',]
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())
       
#categorical columns
cat_cols = ['Gender', 'ParentalSupport', 'ParentalSupport','Online Classes Taken']
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\ntotal null values in the dataset after cleaning :-\n",
      df.isnull().sum())

print("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")

print(df.describe())
