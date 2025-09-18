from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

app = Flask(__name__)
CORS(app)

# Global variable to store model accuracy
MODEL_ACCURACY = None

def train_and_save_model():
    """Train the model and save it as pickle file"""
    global MODEL_ACCURACY
    
    try:
        # Load the Excel dataset
        df = pd.read_excel('diabetes_prediction_dataset.xlsx')
        
        # Features and label
        X = df.drop('diabetes', axis=1)
        y = df['diabetes']
        
        # Encode categorical variables
        X_encoded = pd.get_dummies(X, columns=['gender', 'smoking_history'], drop_first=True)
        
        # Split into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
        
        # Train logistic regression model
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        MODEL_ACCURACY = accuracy_score(y_test, y_pred)
        
        # Save the trained model
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        # Save the feature columns for consistent encoding
        feature_columns = X_encoded.columns.tolist()
        with open('feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_columns, f)
        
        print(f'Model trained successfully! Accuracy: {MODEL_ACCURACY:.2%}')
        return True
        
    except Exception as e:
        print(f'Error training model: {e}')
        return False

def load_model():
    """Load the trained model and feature columns"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        return model, feature_columns
    except:
        return None, None

def prepare_input_data(data, feature_columns):
    """Prepare input data with proper one-hot encoding"""
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([data])
    
    # One-hot encode categorical variables (same as training)
    input_encoded = pd.get_dummies(input_df, columns=['gender', 'smoking_history'], drop_first=True)
    
    # Ensure all feature columns from training are present
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_columns]
    
    return input_encoded

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/accuracy')
def get_accuracy():
    """Return model accuracy"""
    global MODEL_ACCURACY
    if MODEL_ACCURACY is None:
        return jsonify({'accuracy': 'Not available'})
    return jsonify({'accuracy': f'{MODEL_ACCURACY:.2%}'})

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using the trained model"""
    try:
        # Load the model
        model, feature_columns = load_model()
        if model is None:
            return jsonify({'error': 'Model not found. Please train the model first.'})
        
        # Get data from request
        data = request.json
        
        # Validate required fields
        required_fields = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'smoking_history']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'})
        
        # Convert data types
        input_data = {
            'age': float(data['age']),
            'hypertension': int(data['hypertension']),
            'heart_disease': int(data['heart_disease']),
            'bmi': float(data['bmi']),
            'HbA1c_level': float(data['HbA1c_level']),
            'blood_glucose_level': float(data['blood_glucose_level']),
            'gender': data['gender'],
            'smoking_history': data['smoking_history']
        }
        
        # Prepare input data for prediction
        input_encoded = prepare_input_data(input_data, feature_columns)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        prediction_proba = model.predict_proba(input_encoded)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Diabetes' if prediction == 1 else 'No Diabetes',
            'probability': {
                'no_diabetes': f'{prediction_proba[0]:.2%}',
                'diabetes': f'{prediction_proba[1]:.2%}'
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'})

if __name__ == '__main__':
    # Check if model exists, if not train it
    if not os.path.exists('model.pkl'):
        print("Model not found. Training model...")
        if not train_and_save_model():
            print("Failed to train model. Please ensure 'diabetes_prediction_dataset.xlsx' exists.")
            exit(1)
    else:
        # Load accuracy from a separate file if it exists
        try:
            with open('model_accuracy.txt', 'r') as f:
                MODEL_ACCURACY = float(f.read())
        except:
            # If accuracy file doesn't exist, retrain to get accuracy
            train_and_save_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)