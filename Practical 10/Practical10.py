# Step 1 & 2: Import libraries and load dataset
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
dataset = sns.load_dataset('iris')

# Display the first few rows
print("First 5 rows of the dataset:")
print(dataset.head())

# Step 3: Plot Histograms
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.histplot(dataset['sepal_length'], kde=True)
plt.title("Sepal Length Distribution")

plt.subplot(2, 2, 2)
sns.histplot(dataset['sepal_width'], kde=True)
plt.title("Sepal Width Distribution")

plt.subplot(2, 2, 3)
sns.histplot(dataset['petal_length'], kde=True)
plt.title("Petal Length Distribution")

plt.subplot(2, 2, 4)
sns.histplot(dataset['petal_width'], kde=True)
plt.title("Petal Width Distribution")

plt.tight_layout()
plt.show()

# Step 4: Plot Boxplots
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.boxplot(y='sepal_length', x='species', data=dataset)
plt.title("Sepal Length by Species")

plt.subplot(2, 2, 2)
sns.boxplot(y='sepal_width', x='species', data=dataset)
plt.title("Sepal Width by Species")

plt.subplot(2, 2, 3)
sns.boxplot(y='petal_length', x='species', data=dataset)
plt.title("Petal Length by Species")

plt.subplot(2, 2, 4)
sns.boxplot(y='petal_width', x='species', data=dataset)
plt.title("Petal Width by Species")

plt.tight_layout()
plt.show()
