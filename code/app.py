from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
import os 


os.environ['MLFLOW_TRACKING_USERNAME'] = 'admin'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'password'
mlflow.set_tracking_uri("https://mlflow.ml.brain.cs.ait.ac.th")
# Normal Logistic Regression model
model_name_normal = "st125001-a3"
model_version = 1  

# Set up your MLflow tracking URI and load the model
# Ridge logistic Regressionmodel
model_name_ridge = "st125001-a3-ridge"
model_version = 1  

# Load the model from MLflow
# model_a3_normal = mlflow.pyfunc.load_model(f"models:/{model_name_normal}/{model_version}")
# model_a3_ridge = mlflow.pyfunc.load_model(f"models:/{model_name_ridge}/{model_version}")

app = Flask(__name__)

# Load the models and imputers
model_a1 = joblib.load('models/car_price_model.pkl')  # Model A1
model_a2 = joblib.load('models/car_prices_prediction_a2.pkl')  # Model A2
model_a3_normal = joblib.load('models/model_a3_normal.pkl')  # Model A3
model_a3_ridge = joblib.load('models/model_a3_ridge.pkl')  # Model A3
imputer_a1 = joblib.load('models/imputer.pkl')  # Imputer for Model A1
scaler = joblib.load('models/scaler.pkl')
scaler_a3 = joblib.load('models/scaler_a3.pkl')

@app.route('/')
def home():
    return render_template('home.html')  # Home page with links to both models

@app.route('/model_a1')
def model_a1_page():
    return render_template('model_a1.html')  # Form for Model A1

@app.route('/model_a2')
def model_a2_page():
    return render_template('model_a2.html')  # Form for Model A2

@app.route('/model_a3')
def model_a3_page():
    return render_template('model_a3.html')  # Form for Model A2


# Prediction logic for Model A1
@app.route('/predict_a1', methods=['POST'])
def predict_a1():
    input_data = request.get_json()    

    # Prepare data for prediction
    data = {
        'engine': [float(input_data.get('engine'))],
        'max_power': [float(input_data.get('max_power'))],
        'mileage': [float(input_data.get('mileage'))],
        'owner': [int(input_data.get('owner'))]
    }

    features_df = pd.DataFrame(data)    
    features_df = imputer_a1.transform(features_df)    
    feature_names = ['engine', 'max_power', 'mileage', 'owner']
    features_df_imputed = pd.DataFrame(features_df, columns=feature_names)   
    
    try:
        pred_log_price = model_a1.predict(features_df_imputed)
        pred_price = np.exp(pred_log_price)  # Convert log prices back to original scale
        return jsonify({'predicted_price': pred_price[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Prediction logic for Model A2
@app.route('/predict_a2', methods=['POST'])
def predict_a2():
    input_data = request.get_json()

    # Prepare data for prediction
    data2 = {
        'year': [int(input_data.get('year'))],
        'engine': [float(input_data.get('engine'))],
        'max_power': [float(input_data.get('max_power'))],
        'mileage': [float(input_data.get('mileage'))],
        'owner': [int(input_data.get('owner'))]
    }
    features_df2 = pd.DataFrame(data2)
    
    try:
        # Scale the features
        features_scaled = scaler.transform(features_df2)  # Use transform here

        # Make predictions using the model
        pred_log_price = model_a2.predict(features_scaled)

        # Convert log prices back to original scale
        pred_price = np.exp(pred_log_price)

        return jsonify({'predicted_price': pred_price[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# Prediction logic for Model A3
@app.route('/predict_a3', methods=['POST'])
def predict_a3():
    input_data = request.get_json()
    print(f"Received input data for A3: {input_data}")

    # Prepare data for prediction (adjust according to your model input features)
    data = {
        'year': [int(input_data.get('year'))],
        'engine': [float(input_data.get('engine'))],
        'max_power': [float(input_data.get('max_power'))],
        'mileage': [float(input_data.get('mileage'))],
        'owner': [int(input_data.get('owner'))],
        'regression_type': input_data.get('regression_type')
    }
    
    # Debugging the features being passed
    print(f"Prepared features: {data}")    
    
    features_df = pd.DataFrame(data)
    features_to_scale = features_df[['year', 'engine', 'max_power', 'mileage', 'owner']]   
      
    try:
        # Scale only the relevant features
        features_scaled = scaler_a3.transform(features_to_scale)  
        print(f"Scaled features: {features_scaled}")

        # Create a new DataFrame for scaled features
        features_scaled_df = pd.DataFrame(features_scaled, columns=['year', 'engine', 'max_power', 'mileage', 'owner'])
        
        # Add the 'owner' feature back to the DataFrame
        # features_scaled_df['owner'] = features_df['owner'].values  # Re-add 'owner' to the DataFrame
        
        # features_numpy = features_scaled_df.to_numpy()
        
        #Determine which model to use based on input
        regression_type = input_data.get('regression_type')
        if regression_type == 'normal':
            pred_class = model_a3_normal.predict(features_scaled_df)
        elif regression_type == 'ridge':
            pred_class = model_a3_ridge.predict(features_scaled_df)
        else:
            return jsonify({'error': 'Invalid model type specified. Use "normal" or "ridge".'}), 400               
        
        # Prepare the response
        return jsonify({'predicted_class': int(pred_class[0])})    
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error
        return jsonify({'error': str(e)}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
