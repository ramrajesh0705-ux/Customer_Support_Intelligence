from pathlib import Path
import joblib
import pandas as pd
from typing import Dict
import warnings
from .data import build_ticket_text

# Suppress sklearn version warnings that don't affect functionality
warnings.filterwarnings('ignore', category=UserWarning, message='.*Trying to unpickle estimator.*')

MODEL_FILENAMES = {
    'ticket_type': 'ticket_type.joblib',
    'ticket_priority': 'ticket_priority.joblib',
    'resolution_time': 'resolution_time.joblib',
}


class TicketInference:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        try:
            self.ticket_type_model = joblib.load(self.model_dir / MODEL_FILENAMES['ticket_type'])
            self.priority_model = joblib.load(self.model_dir / MODEL_FILENAMES['ticket_priority'])
            self.resolution_model = joblib.load(self.model_dir / MODEL_FILENAMES['resolution_time'])
        except Exception as e:
            raise RuntimeError(f'Failed to load models from {self.model_dir}: {e}')

    def _prepare_input(self, payload: Dict) -> pd.DataFrame:
        item = {
            'ticket_subject': payload.get('ticket_subject', ''),
            'ticket_description': payload.get('ticket_description', ''),
            'ticket_channel': payload.get('ticket_channel', 'Email'),
            'product_purchased': payload.get('product_purchased', 'Cloud CRM'),
            'customer_age': float(payload.get('customer_age', 0)),
            'customer_gender': payload.get('customer_gender', 'Other'),
            'first_response_time': float(payload.get('first_response_time', 0.0)),
        }
        item['ticket_text'] = item['ticket_subject'] + ' ' + item['ticket_description']
        return pd.DataFrame([item])

    def predict(self, payload: Dict) -> Dict:
        input_df = self._prepare_input(payload)
        ticket_type = self.ticket_type_model.predict(input_df)[0]
        ticket_priority = self.priority_model.predict(input_df)[0]
        time_to_resolution = float(self.resolution_model.predict(input_df)[0])

        return {
            'ticket_type': ticket_type,
            'ticket_priority': ticket_priority,
            'time_to_resolution_hours': round(time_to_resolution, 2),
        }
