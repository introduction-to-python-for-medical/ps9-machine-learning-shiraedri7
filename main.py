import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('parkinsons.csv')
df.head()

# Select features
X = df[['PPE', 'RPDE']]
y = df['status']

# Create pair plots
sns.pairplot(df, vars=['PPE', 'RPDE', 'status'], hue='status')
plt.show()

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the input features
X_scaled = scaler.fit_transform(X)

# Convert the scaled data back to a DataFrame (optional)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
X_scaled

from sklearn.model_selection import train_test_split

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42) # You can adjust n_estimators
rf_classifier.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = rf_classifier.predict(X_val)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}"

import joblib

# Save the model
model_filename = 'parkinsons_model.joblib'
joblib.dump(rf_classifier, model_filename)

