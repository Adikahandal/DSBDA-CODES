# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ------------------------
# Part 1: Simple Regression using polyfit
# ------------------------

# 1. Create arrays
x = np.array([95, 85, 80, 70, 60])
y = np.array([85, 95, 70, 65, 70])

# 2. Linear regression using polyfit
model = np.polyfit(x, y, 1)
print("Polyfit Model Coefficients:", model)

# 3. Predict y for a given x value
predict = np.poly1d(model)
print("Prediction for x=65:", predict(65))

# 4. Predict for all x values
y_pred = predict(x)
print("All Predicted y values:", y_pred)

# 5. R-squared score
print("R-squared Score:", r2_score(y, y_pred))

# 6. Plot the regression
y_line = model[1] + model[0] * x
plt.plot(x, y_line, c='red', label='Regression Line')
plt.scatter(x, y_pred, label='Predicted', color='green')
plt.scatter(x, y, c='blue', label='Actual')
plt.xlabel("X values")
plt.ylabel("Y values")
plt.title("Simple Linear Regression using Numpy")
plt.legend()
plt.show()

# ------------------------
# Part 2: Boston Housing Dataset Regression
# ------------------------

# 1. Load Boston housing dataset (from CSV, e.g. Kaggle)
# Replace the path below with your local CSV path
data = pd.read_csv(r'C:\IMP NOTES\Practical 4\housing.csv')

# 2. Check for missing values
print("\nMissing values:\n", data.isnull().sum())

# 3. Define independent and dependent variables
X = data.drop(['MEDV'], axis=1)
y = data['MEDV']

# 4. Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

# 5. Linear Regression Model
lm = LinearRegression()
lm.fit(xtrain, ytrain)

# 6. Predictions
ytrain_pred = lm.predict(xtrain)
ytest_pred = lm.predict(xtest)

# 7. Evaluate performance
print("\nTraining MSE:", mean_squared_error(ytrain, ytrain_pred))
print("Testing MSE:", mean_squared_error(ytest, ytest_pred))
print("Training R2 Score:", r2_score(ytrain, ytrain_pred))
print("Testing R2 Score:", r2_score(ytest, ytest_pred))

# 8. Plotting
plt.scatter(ytrain, ytrain_pred, c='blue', marker='o', label='Training data')
plt.scatter(ytest, ytest_pred, c='lightgreen', marker='s', label='Test data')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title("True vs Predicted Values - Boston Housing")
plt.legend(loc='upper left')
plt.grid()
plt.show()
