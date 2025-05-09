import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('titanic')

# Display first 5 rows
print(df.head())

# Handle missing values for meaningful boxplot visualization
df['age'] = df['age'].fillna(df['age'].median())

# Basic Boxplot: Age distribution based on Sex
plt.figure(figsize=(6, 4))
sns.boxplot(x='sex', y='age', data=df)
plt.title('Boxplot of Age vs Sex')
plt.show()

# Boxplot with hue: Add Survival Status
plt.figure(figsize=(6, 4))
sns.boxplot(x='sex', y='age', data=df, hue='survived')
plt.title('Boxplot of Age vs Sex (Hue: Survived)')
plt.show()
