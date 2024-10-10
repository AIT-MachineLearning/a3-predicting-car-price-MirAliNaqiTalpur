import unittest
import json
from app import app  # Import your Flask app

class TestPredictA3(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_a3(self):
        # Minimal input data that conforms to the expected format for Model A3
        input_data = {
            'year': 2020,
            'engine': 1.5,
            'max_power': 150,
            'mileage': 15.5,
            'owner': 0,
            'regression_type': 'normal'  # Change to 'ridge' to test that case
        }

        # Send POST request to the predict_a3 endpoint
        response = self.app.post('/predict_a3',
                                 data=json.dumps(input_data),
                                 content_type='application/json')

        # Check if the status code is 200 OK
        self.assertEqual(response.status_code, 200)

        # Verify the response contains the 'predicted_class' field
        response_data = response.get_json()
        self.assertIn('predicted_class', response_data)

        # Optional: Check if the predicted_class is an integer
        self.assertIsInstance(response_data['predicted_class'], int)

if __name__ == '__main__':
    unittest.main()
