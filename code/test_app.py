import unittest
import json
from app import app  # Import your Flask app

class TestPredictA3(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_model_prediction(self):
        # Input data with 'regression_type' included
        input_data = {
            'year': 2020,
            'engine': 1.5,
            'max_power': 150,
            'mileage': 15.5,
            'owner': 0,
            'regression_type': 'normal'  # Add this field
        }

        # Make a POST request and check status code
        response = self.app.post('/predict_a3',
                                 data=json.dumps(input_data),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)  # Expecting HTTP 200 OK
        response_data = json.loads(response.data)
        self.assertIn('predicted_class', response_data)
        self.assertIsInstance(response_data['predicted_class'], int)

if __name__ == '__main__':
    unittest.main()
