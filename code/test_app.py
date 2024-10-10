# test_app.py
import unittest
import json
from app import app  # Adjust the import based on your app's structure

class PredictA3TestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_a3_normal(self):
        input_data = {
            'year': 2020,
            'engine': 1.5,
            'max_power': 120.0,
            'mileage': 15.0,
            'owner': 1,
            'regression_type': 'normal'
        }
        response = self.app.post('/predict_a3', json=input_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predicted_class', data)

    def test_predict_a3_ridge(self):
        input_data = {
            'year': 2020,
            'engine': 1.5,
            'max_power': 120.0,
            'mileage': 15.0,
            'owner': 1,
            'regression_type': 'ridge'
        }
        response = self.app.post('/predict_a3', json=input_data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('predicted_class', data)

    def test_predict_a3_invalid_model(self):
        input_data = {
            'year': 2020,
            'engine': 1.5,
            'max_power': 120.0,
            'mileage': 15.0,
            'owner': 1,
            'regression_type': 'invalid_model'
        }
        response = self.app.post('/predict_a3', json=input_data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()
