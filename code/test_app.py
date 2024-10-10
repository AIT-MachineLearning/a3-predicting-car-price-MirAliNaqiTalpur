import pytest
import json
from app import app

# Create a test client using the Flask app
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Sample input data for testing
@pytest.fixture
def valid_input_data():
    return {
        'year': 2015,
        'engine': 1500.0,
        'max_power': 110.0,
        'mileage': 15.0,
        'owner': 1,
        'regression_type': 'normal'
    }

# Test 1: Check if the model accepts the expected input
def test_predict_a3_input(client, valid_input_data):
    response = client.post('/predict_a3', 
                           data=json.dumps(valid_input_data), 
                           content_type='application/json')

    # Assert that the response status code is 200 (successful request)
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}"

# Test 2: Check if the model output has the expected shape
def test_predict_a3_output_shape(client, valid_input_data):
    response = client.post('/predict_a3', 
                           data=json.dumps(valid_input_data), 
                           content_type='application/json')

    assert response.status_code == 200, f"Expected 200 but got {response.status_code}"
    response_json = response.get_json()
    assert 'predicted_class' in response_json, "The 'predicted_class' field is missing in the response"
    assert isinstance(response_json['predicted_class'], int), "The 'predicted_class' should be an integer"
    assert 0 <= response_json['predicted_class'] <= 100, "The 'predicted_class' should be within the expected range (0-100)"

# Parameterized test for invalid inputs
@pytest.mark.parametrize("input_data, expected_status", [
    ({'year': 2025, 'engine': 1500.0, 'max_power': 110.0, 'mileage': 15.0, 'owner': 1, 'regression_type': 'normal'}, 400),  # Future year
    ({'year': 2015, 'engine': -1500.0, 'max_power': 110.0, 'mileage': 15.0, 'owner': 1, 'regression_type': 'normal'}, 400),  # Negative engine
    ({'year': 2015, 'engine': 1500.0, 'max_power': 110.0, 'mileage': 15.0, 'regression_type': 'normal'}, 400)  # Missing owner
])
def test_invalid_input(client, input_data, expected_status):
    response = client.post('/predict_a3', 
                           data=json.dumps(input_data), 
                           content_type='application/json')
    
    assert response.status_code == expected_status, f"Expected {expected_status} but got {response.status_code}"
