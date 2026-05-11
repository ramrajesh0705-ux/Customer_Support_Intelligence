from pathlib import Path
import joblib
import pandas as pd
from typing import Dict
from huggingface_hub import hf_hub_download
from .data import build_ticket_text

MODEL_REPO = "rajesh7593/customersupportintelligence-models"

MODEL_FILENAMES = {
    'ticket_type': 'ticket_type.joblib',
    'ticket_priority': 'ticket_priority.joblib',
    'resolution_time': 'resolution_time.joblib',
}


class TicketInference:
    def __init__(self, model_dir: Path = None):
        self.ticket_type_model = joblib.load(hf_hub_download(MODEL_REPO, MODEL_FILENAMES['ticket_type']))
        self.priority_model = joblib.load(hf_hub_download(MODEL_REPO, MODEL_FILENAMES['ticket_priority']))
        self.resolution_model = joblib.load(hf_hub_download(MODEL_REPO, MODEL_FILENAMES['resolution_time']))

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
