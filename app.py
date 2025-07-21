# app.py - Flask Web Application for Student Performance Prediction
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
try:
    with open('student_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    feature_names = model_data['feature_names']
    
    print("Model loaded successfully!")
except:
    model = None
    print("Model not found. Please train the model first.")

@app.route('/')
def home():
    """Home page with prediction form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'})
        
        # Get form data
        form_data = request.form.to_dict()
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([form_data])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                input_data[col + '_encoded'] = encoder.transform([input_data[col][0]])[0]
            else:
                input_data[col + '_encoded'] = 0  # Default if not provided
        
        # Convert numeric columns
        numeric_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 
                          'failures', 'famrel', 'freetime', 'goout', 'Dalc', 
                          'Walc', 'health', 'absences']
        
        for col in numeric_columns:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce').fillna(0)
        
        # Prepare features in the same order as training
        X_input = pd.DataFrame({feature: [0] for feature in feature_names})
        for feature in feature_names:
            if feature in input_data.columns:
                X_input.at[0, feature] = input_data[feature].iloc[0]
            elif feature in input_data:
                X_input.at[0, feature] = input_data[feature]

        # Ensure correct order and type
        X_input = X_input[feature_names]
        X_input = X_input.astype(float)
        
        # Make prediction
        if hasattr(model, 'feature_importances_'):
            # Random Forest - no scaling needed
            prediction = model.predict(X_input)[0]
        else:
            # Linear Regression - scaling needed
            X_scaled = scaler.transform(X_input)
            prediction = model.predict(X_scaled)[0]
        
        # Round prediction
        prediction = round(prediction, 2)
        
        # Determine grade category
        if prediction >= 16:
            grade_category = "Excellent"
            color = "success"
        elif prediction >= 14:
            grade_category = "Good"
            color = "info"
        elif prediction >= 10:
            grade_category = "Satisfactory"
            color = "warning"
        else:
            grade_category = "Needs Improvement"
            color = "danger"
        
        return jsonify({
            'prediction': prediction,
            'grade_category': grade_category,
            'color': color
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch predictions via API"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'})
        
        data = request.get_json()
        predictions = []
        
        for student_data in data:
            # Process similar to single prediction
            # ... (implement batch processing logic)
            pass
        
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)