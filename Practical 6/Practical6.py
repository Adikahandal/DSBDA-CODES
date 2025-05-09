# Step 1: Import libraries and create aliases for Pandas, Numpy, and Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score, recall_score

# Step 2: Load the Social Network Ads dataset
# Assuming the dataset is in CSV format, you can load it like this:
df = pd.read_csv(r"C:\IMP NOTES\Practical 6\Social_Network_Ads.csv")

# Step 3: Inspect the first few rows of the dataset
print(df.head())

# Step 4: Perform Data Preprocessing
# ● Convert Categorical to Numerical Values if applicable (If you have categorical data)
# ● Check for Null values
print(df.isnull().sum())  # Check for null values

# Step 5: Divide the dataset into Independent (X) and Dependent (Y) variables
X = df[['Age', 'EstimatedSalary']]  # Independent variables (features)
y = df['Purchased']  # Dependent variable (target)

# Step 6: Split the dataset into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 7: Scale the features (important for Naive Bayes)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 8: Train the Naive Bayes model
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

# Step 9: Predict the y_pred for the test set
y_pred = gaussian.predict(X_test)

# Step 10: Evaluate the performance of the model for train_y and test_y
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Step 11: Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
