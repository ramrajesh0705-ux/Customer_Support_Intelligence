from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.customer_support_intelligence.inference import TicketInference

app = FastAPI(title='Customer Support Intelligence API')

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'

class TicketRequest(BaseModel):
    ticket_subject: str = Field(..., example='Unable to access billing page')
    ticket_description: str = Field(..., example='I cannot update my credit card details on the portal.')
    ticket_channel: str = Field('Email', example='Email')
    product_purchased: str = Field('Cloud CRM', example='Cloud CRM')
    customer_age: int = Field(30, example=30)
    customer_gender: str = Field('Female', example='Female')
    date_of_purchase: str = Field('2025-08-15', example='2025-08-15')
    first_response_time: float = Field(1.2, example=1.2)

@app.on_event('startup')
def load_models():
    global predictor
    try:
        predictor = TicketInference(MODEL_DIR)
    except Exception as exc:
        raise RuntimeError(f'Failed to load models: {exc}')

@app.get('/health')
def health_check():
    return {'status': 'ok'}

@app.post('/predict')
def predict(request: TicketRequest):
    try:
        result = predictor.predict(request.dict())
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
