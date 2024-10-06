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
    
if __name__ == "__main__":
    pytest.main()
