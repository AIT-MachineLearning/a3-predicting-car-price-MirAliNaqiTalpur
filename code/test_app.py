import json
import pytest
from app import app  # Assuming your Flask app is in app.py

# Initialize the Flask application for testing
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test the home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Welcome' in response.data  # Adjust based on your home page content

def test_model_a1_prediction(client):
    """Test predictions for Model A1."""
    input_data = {
        'engine': 1500,
        'max_power': 100,
        'mileage': 15.0,
        'owner': 0
    }
    response = client.post('/predict_a1', data=json.dumps(input_data), content_type='application/json')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predicted_price' in json_data

def test_model_a2_prediction(client):
    """Test predictions for Model A2."""
    input_data = {
        'year': 2020,
        'engine': 1500,
        'max_power': 100,
        'mileage': 15.0,
        'owner': 0
    }
    response = client.post('/predict_a2', data=json.dumps(input_data), content_type='application/json')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predicted_price' in json_data

def test_model_a3_normal_prediction(client):
    """Test predictions for Model A3 (normal regression)."""
    input_data = {
        'year': 2020,
        'engine': 1500,
        'max_power': 100,
        'mileage': 15.0,
        'owner': 0,
        'regression_type': 'normal'
    }
    response = client.post('/predict_a3', data=json.dumps(input_data), content_type='application/json')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predicted_class' in json_data

def test_model_a3_ridge_prediction(client):
    """Test predictions for Model A3 (ridge regression)."""
    input_data = {
        'year': 2020,
        'engine': 1500,
        'max_power': 100,
        'mileage': 15.0,
        'owner': 0,
        'regression_type': 'ridge'
    }
    response = client.post('/predict_a3', data=json.dumps(input_data), content_type='application/json')
    assert response.status_code == 200
    json_data = response.get_json()
    assert 'predicted_class' in json_data

def test_invalid_model_type(client):
    """Test prediction for Model A3 with an invalid regression type."""
    input_data = {
        'year': 2020,
        'engine': 1500,
        'max_power': 100,
        'mileage': 15.0,
        'owner': 0,
        'regression_type': 'invalid'  # Invalid type to test error handling
    }
    response = client.post('/predict_a3', data=json.dumps(input_data), content_type='application/json')
    assert response.status_code == 400
    json_data = response.get_json()
    assert 'error' in json_data
    assert json_data['error'] == 'Invalid model type specified. Use "normal" or "ridge".'

if __name__ == "__main__":
    pytest.main()
