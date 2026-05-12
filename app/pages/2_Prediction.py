import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
from pathlib import Path
from src.customer_support_intelligence.inference import TicketInference

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / 'models'

@st.cache_resource
def load_predictor():
    return TicketInference(MODEL_DIR)

predictor = load_predictor()

st.title('Customer Support Intelligence - Prediction')

ticket_subject = st.text_input('Ticket Subject', 'Unable to access billing page')
ticket_description = st.text_area('Ticket Description', 'I cannot update my credit card details on the portal.')
ticket_channel = st.selectbox('Ticket Channel', ['Email', 'Phone', 'Chat'])
product_purchased = st.selectbox('Product Purchased', ['Cloud CRM', 'ERP System', 'HR Software'])
customer_age = st.number_input('Customer Age', min_value=18, max_value=100, value=30)
customer_gender = st.selectbox('Customer Gender', ['Male', 'Female', 'Other'])
date_of_purchase = st.date_input('Date of Purchase')
first_response_time = st.number_input('First Response Time (hours)', min_value=0.0, value=1.2)

if st.button('Predict'):
    sample = {
        'ticket_subject': ticket_subject,
        'ticket_description': ticket_description,
        'ticket_channel': ticket_channel,
        'product_purchased': product_purchased,
        'customer_age': customer_age,
        'customer_gender': customer_gender,
        'date_of_purchase': str(date_of_purchase),
        'first_response_time': first_response_time,
    }
    result = predictor.predict(sample)
    st.write('Prediction Result:')
    st.json(result)