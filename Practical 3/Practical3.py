import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing

# Load the dataset
df = pd.read_csv(r"C:\IMP NOTES\Practical 3\tmdb_5000_credits.csv")
df.columns = df.columns.str.strip().str.lower()

print("\n=== NUMERICAL COLUMN ANALYSIS (Only 'movie_id') ===")

# Column-wise Mean
print("\n Mean (Column-wise):")
print(df.mean(numeric_only=True).to_string())

# Row-wise Mean
print("\n Mean (Row-wise for first 4 rows):")
print(df.mean(axis=1, numeric_only=True).head(4).to_string())

# Column-wise Median
print("\n Median (Column-wise):")
print(df.median(numeric_only=True).to_string())

# Row-wise Median
print("\n Median (Row-wise for first 4 rows):")
print(df.median(axis=1, numeric_only=True).head(4).to_string())

# Mode
print("\n Mode (Column-wise):")
print(df.mode(numeric_only=True).iloc[0].to_string())

# Minimum
print("\n Minimum (Column-wise):")
print(df.min(numeric_only=True).to_string())

# Maximum
print("\n Maximum (Column-wise):")
print(df.max(numeric_only=True).to_string())

# Standard Deviation
print("\n Standard Deviation (Column-wise):")
print(df.std(numeric_only=True).to_string())
