#Import the Dependencies

import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import matplotlib.pyplot as plt

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("mountains.csv")

data.head()

pd.set_option('display.max_columns', None)

data.head()

#Check for missing values
data.isnull().sum()

#Check for duplicate values
data[data.duplicated()]

#Removing the Outliers using BoxPlot
def zohaib (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["Age"])

data = zohaib(data,"Age")

#Changing Characters into numerics
#Label Encoding

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(data["Gender"])

data["Gender"] = encoder.transform(data["Gender"])

data["Gender"].unique()

data.head()

data.info()


#Automating the EDA using AutoViz
from autoviz.AutoViz_Class import AutoViz_Class 
AV = AutoViz_Class()
import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'mountains.csv'
sep =","
dft = AV.AutoViz(
    filename  
)

#Scale the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#The dataclass is imbalance, I have checked the class using a function
def class_distribution(data, column_name='Preference'):
    # Display total counts and percentage for each class
    distribution = data[column_name].value_counts()
    percentage = data[column_name].value_counts(normalize=True) * 100
    
    print(f"Class distribution for '{column_name}':")
    print(distribution)
    print("\nPercentage distribution:")
    print(percentage)

# Call the function to display the distribution for the 'Resigned' column
class_distribution(data, 'Preference')

#Using the Smote process to resample the dataclass
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
smote = SMOTE(random_state=42) 
X_resampled,y_resampled = smote.fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3, random_state = 20)

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print("Before SMOTE: ", X_train.shape, y_train.shape)
print("After SMOTE: ", X_train_over.shape, y_train_over.shape)
print("After SMOTE Label Distribution: ", pd.Series(y_train_over).value_counts())


#Segregating the dataset into X and Y
#Segregrating dataset into X and y

X = data.drop("Preference", axis = 1)

y = data["Preference"]

X.head()

y.head()

#Splitting the dataset into testing and training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Calling the machine learning algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# initialize classifiers
log_model = LogisticRegression()
rf_model = RandomForestClassifier()
knn_model = KNeighborsClassifier()
gnb_model = GaussianNB()

# logistic regression model fitting
log_model.fit(X_train, y_train)

# random forest classifier model fitting
rf_model.fit(X_train, y_train)

# k nearest neighbour model fitting
knn_model.fit(X_train, y_train)

# gaussian naive bayes model fitting
gnb_model.fit(X_train, y_train)


# model testing libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# logistic regression model testing
y_pred_log = log_model.predict(X_test)
print("Logistic Regression")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")
print(f"Precision: {precision_score(y_test, y_pred_log, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_log, average='weighted')}")
print(f"F1 score: {f1_score(y_test, y_pred_log, average='weighted')}")



































































































