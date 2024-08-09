import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('data/cardio_train.csv', delimiter=';')

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())

# Basic Data Preprocessing
# Handle missing values (if any)
print("\nChecking for missing values:")
print(data.isnull().sum())

# Feature engineering (if necessary, e.g., converting age to years)
data['age_years'] = data['age'] / 365

# Drop the 'id' column if it's not needed
data = data.drop(columns=['id'])

# Visualization
# Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()

# Distribution of the target variable (cardio)
plt.figure(figsize=(6, 4))
sns.countplot(x='cardio', data=data)
plt.title('Distribution of Target Variable (Cardio)')
plt.show()

# Pairplot to visualize relationships between features (optional, can be slow with large datasets)
# sns.pairplot(data, hue='cardio')
# plt.show()

# Data Preprocessing for Machine Learning
X = data.drop(columns=['cardio'])
y = data['cardio']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (important for algorithms like SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize machine learning models
models = {
    'Support Vector Machine': SVC(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Save the best model (optional)
# import joblib
# joblib.dump(models['Random Forest'], 'best_model.pkl')

