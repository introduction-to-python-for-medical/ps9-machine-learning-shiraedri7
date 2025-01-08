import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import yaml

# Load the dataset
df = pd.read_csv('parkinsons.csv')
df.head()

# Select features
X = df[['PPE', 'RPDE']]
y = df['status']

# Create pair plots
sns.pairplot(df, vars=['PPE', 'RPDE', 'status'], hue='status')
plt.show()

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)  # You can tune hyperparameters here if needed
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy}")

# Save the model
model_filename = 'parkinsons_model.joblib'
joblib.dump(clf, model_filename)

# Define the configuration
config = {
    'selected_features': ['PPE', 'RPDE'],
    'path': 'parkinsons_model.joblib'
}

# Write the configuration to a YAML file
with open('config.yaml', 'w') as file:
    yaml.dump(config, file)
