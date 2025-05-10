import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import zscore

# Load the dataset
missing_values = ["na", "Na", "--", " ", "NaN", ""]
df = pd.read_csv(r"C:\IMP NOTES\Practical 2\StudentsPerformanceTest1.tsv", 
                 na_values=missing_values, 
                 delimiter='\t')

print("Original Dataset:\n", df)

# Show missing values count
print("\nMissing Values:\n", df.isnull().sum())

# Fill missing numeric values with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Drop rows with missing categorical data (optional)
df.dropna(subset=df.select_dtypes(include=object).columns, inplace=True)

# Handle outliers using Z-score
numeric_cols = df.select_dtypes(include=np.number).columns
z_scores = zscore(df[numeric_cols])
df = df[(np.abs(z_scores) < 3).all(axis=1)]

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include=object).columns:
    df[col + "_encoded"] = le.fit_transform(df[col])

print("\nFinal Cleaned and Encoded Dataset:\n", df)
