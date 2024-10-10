import unittest
import json
from app import app  # Import your Flask app

class TestPredictA3(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_model_takes_expected_input(self):
        # Prepare the input data
        input_data = {
            'year': 2020,
            'engine': 1.5,
            'max_power': 150,
            'mileage': 15.5,
            'owner': 0,
            'regression_type': 'normal'
        }
        
        # Make a POST request to the endpoint
        response = self.app.post('/predict_a3', 
                                  data=json.dumps(input_data),
                                  content_type='application/json')
        
        # Check that the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)
        
        # Check the response data
        response_data = json.loads(response.data)
        self.assertIn('predicted_class', response_data)

    def test_model_output_shape(self):
        # Prepare the input data
        input_data = {
            'year': 2020,
            'engine': 1.5,
            'max_power': 150,
            'mileage': 15.5,
            'owner': 0,
            'regression_type': 'normal'
        }
        
        # Make a POST request to the endpoint
        response = self.app.post('/predict_a3', 
                                  data=json.dumps(input_data),
                                  content_type='application/json')
        
        # Check that the response is successful (status code 200)
        self.assertEqual(response.status_code, 200)
        
        # Check the output shape (expected to be a single predicted class)
        response_data = json.loads(response.data)
        self.assertIsInstance(response_data['predicted_class'], int)

if __name__ == '__main__':
    unittest.main()
