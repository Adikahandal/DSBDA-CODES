import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
df = sns.load_dataset('titanic')

# Display first few rows
print(df.head())

# -------------------------
# Data Preprocessing
# -------------------------

# Fill missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Add 'Unknown' category to 'deck' before filling
df['deck'] = df['deck'].cat.add_categories('Unknown')
df['deck'].fillna('Unknown', inplace=True)

# -------------------------
# Visualization
# -------------------------

# 1. Boxplot of age by sex
plt.figure(figsize=(8, 4))
sns.boxplot(x='sex', y='age', data=df)
plt.title("Boxplot of Age by Sex")
plt.show()

# 2. Boxplot of age by sex and survival
plt.figure(figsize=(8, 4))
sns.boxplot(x='sex', y='age', hue='survived', data=df)
plt.title("Boxplot of Age by Sex and Survival")
plt.show()

# 3. Countplot of survival
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()

# 4. Heatmap of correlation
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 5. Histogram of age distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.show()

# -------------------------
# Basic Analysis
# -------------------------

# Survival rate by class
print("\nSurvival rate by class:")
print(df.groupby('pclass')['survived'].mean())

# Survival rate by gender
print("\nSurvival rate by gender:")
print(df.groupby('sex')['survived'].mean())

# Average fare by class
print("\nAverage fare by class:")
print(df.groupby('pclass')['fare'].mean())


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# df = sns.load_dataset('titanic')

# # Handle missing values
# df['age'] = df['age'].fillna(df['age'].median())
# df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
# df['deck'] = df['deck'].cat.add_categories('Unknown').fillna('Unknown')

# # 1. Boxplot → Distribution of Age by Sex (Categorical vs Numerical)
# plt.figure(figsize=(6, 4))
# sns.boxplot(x='sex', y='age', data=df)
# plt.title("Boxplot: Age vs Sex")
# plt.show()

# # 2. Countplot → Count of passengers in each class (Categorical)
# plt.figure(figsize=(6, 4))
# sns.countplot(x='class', data=df)
# plt.title("Countplot: Passengers per Class")
# plt.show()

# # 3. Scatterplot → Age vs Fare (Numerical vs Numerical)
# plt.figure(figsize=(6, 4))
# sns.scatterplot(x='age', y='fare', data=df, hue='survived')
# plt.title("Scatterplot: Age vs Fare (colored by survival)")
# plt.show()

# # 4. Heatmap → Correlation matrix (Multivariate Numeric)
# plt.figure(figsize=(6, 4))
# sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
# plt.title("Heatmap: Feature Correlation")
# plt.show()
