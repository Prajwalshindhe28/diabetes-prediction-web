import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Excel dataset
df = pd.read_excel('diabetes_prediction_dataset.xlsx')

# Features and label
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Encode categorical variables if needed (e.g., gender, smoking_history)
# Converting categorical columns to numeric using one-hot encoding
X = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=True)

# Split into training and testing data (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2%}')
