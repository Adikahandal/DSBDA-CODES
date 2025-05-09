# Step 1: Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Step 2: Load the dataset
df = pd.read_csv(r"C:\IMP NOTES\Practical 5\Social_Network_Ads.csv")

# Step 3: Display the 'Gender' column
print("\nGender Column:\n", df['Gender'].head())

# Step 4: Data Preprocessing

# a. Check data types
print("\nData Types:\n", df.dtypes)

# b. Map Gender to numerical values
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
print("\nMapped Gender Column:\n", df['Gender'].head())

# c. Check for null values
print("\nNull Value Check:\n", df.isnull().sum())

# d. Covariance Matrix (optional analysis)
print("\nCovariance Matrix:\n", df.cov(numeric_only=True))

# e. Split into independent and dependent variables
x = df.drop(['Age'], axis=1)
y = df['Age']

# f. Split dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25, random_state=0)

# g. Scale features
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# Step 5: Train Logistic Regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(xtrain, ytrain)

# Step 6: Predict on test set
y_pred = classifier.predict(xtest)
print("\nPredicted Y values:\n", y_pred)

# Step 7 & 8: Evaluate performance
cm = confusion_matrix(ytest, y_pred)
print("\nConfusion Matrix:\n", cm)
