import pytest
import json
from app import app

# Create a test client using the Flask app
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

# Test 1: Check if the model accepts the expected input
def test_predict_a3_input(client):
    input_data = {
        'year': 2015,
        'engine': 1500.0,
        'max_power': 110.0,
        'mileage': 15.0,
        'owner': 1,
        'regression_type': 'normal'
    }

    response = client.post('/predict_a3', 
                           data=json.dumps(input_data), 
                           content_type='application/json')

    # Assert that the response status code is 200 (successful request)
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}"

# Test 2: Check if the model output has the expected shape
def test_predict_a3_output_shape(client):
    input_data = {
        'year': 2015,
        'engine': 1500.0,
        'max_power': 110.0,
        'mileage': 15.0,
        'owner': 1,
        'regression_type': 'normal'
    }

    response = client.post('/predict_a3', 
                           data=json.dumps(input_data), 
                           content_type='application/json')

    # Assert that the response status code is 200 (successful request)
    assert response.status_code == 200, f"Expected 200 but got {response.status_code}"

    # Parse the JSON response
    response_json = response.get_json()

    # Assert that the 'predicted_class' field is present in the response
    assert 'predicted_class' in response_json, "The 'predicted_class' field is missing in the response"

    # Assert that the output has the expected shape (in this case, it should be an integer)
    assert isinstance(response_json['predicted_class'], int), "The 'predicted_class' should be an integer"
