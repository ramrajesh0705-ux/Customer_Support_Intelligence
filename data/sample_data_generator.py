import argparse
import random
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta

TICKET_TYPES = ['Billing', 'Technical', 'Account', 'Product Inquiry']
PRIORITIES = ['Low', 'Medium', 'High', 'Critical']
CHANNELS = ['Email', 'Chat', 'Social Media', 'Phone']
PRODUCTS = ['Cloud CRM', 'Analytics Suite', 'Marketing Automation', 'Payments Platform']
GENDERS = ['Female', 'Male', 'Other']
STATUS = ['Open', 'Pending Customer Response', 'Closed']

SUBJECT_TEMPLATES = {
    'Billing': ['Invoice issue on my account', 'Overcharged for subscription', 'Refund request needed'],
    'Technical': ['Login page fails to load', 'App crashes on startup', 'API timeout error'],
    'Account': ['Need help resetting password', 'Cannot access account settings', 'Profile update issue'],
    'Product Inquiry': ['Question about product features', 'How to use the dashboard', 'Clarification on pricing tiers'],
}

DESCRIPTION_TEMPLATES = {
    'Billing': [
        'I was billed twice for this month and I need a correction.',
        'The invoice amount does not match the plan I selected.',
        'Please issue a refund for the unexpected charge on my card.',
    ],
    'Technical': [
        'When I try to login, the page hangs and eventually returns an error.',
        'The application crashes after I upload a file and I cannot continue.',
        'The API response is failing with timeout and I need assistance.',
    ],
    'Account': [
        'I cannot change my password because the link is invalid.',
        'My account settings are not saving after I update them.',
        'I need help confirming my email address and finishing setup.',
    ],
    'Product Inquiry': [
        'Does the new dashboard support custom reporting widgets?',
        'What is the difference between the standard and premium plan?',
        'Can I integrate this product with our existing CRM provider?',
    ],
}

RESOLUTIONS = {
    'Billing': ['Issued a refund and corrected the invoice.', 'Updated payment method and clarified charges.'],
    'Technical': ['Restarted service and patched the failing endpoint.', 'Reset credentials and verified the connection.'],
    'Account': ['Assisted with password reset and verified account settings.', 'Confirmed email and resolved the access issue.'],
    'Product Inquiry': ['Shared product documentation and next steps.', 'Explained pricing tiers and feature list.'],
}


def generate_row(index: int) -> dict:
    ticket_type = random.choice(TICKET_TYPES)
    subject = random.choice(SUBJECT_TEMPLATES[ticket_type])
    description = random.choice(DESCRIPTION_TEMPLATES[ticket_type])
    priority = random.choices(PRIORITIES, weights=[0.35, 0.3, 0.2, 0.15], k=1)[0]
    channel = random.choice(CHANNELS)
    product = random.choice(PRODUCTS)
    gender = random.choice(GENDERS)
    age = random.randint(18, 70)
    purchase_date = datetime.now() - timedelta(days=random.randint(5, 365))
    response_time = round(random.uniform(0.2, 8.0), 1)
    resolution_hours = max(0.5, round(random.gauss(12 if priority in ['High', 'Critical'] else 6, 4), 1))
    satisfaction = random.randint(1, 5)
    status = random.choice(STATUS)
    resolution = random.choice(RESOLUTIONS[ticket_type])

    return {
        'ticket_id': f'TKT-{index:06d}',
        'customer_name': f'Customer {index}',
        'customer_email': f'user{index}@example.com',
        'customer_age': age,
        'customer_gender': gender,
        'product_purchased': product,
        'date_of_purchase': purchase_date.strftime('%Y-%m-%d'),
        'ticket_type': ticket_type,
        'ticket_subject': subject,
        'ticket_description': description,
        'ticket_status': status,
        'resolution': resolution,
        'ticket_priority': priority,
        'ticket_channel': channel,
        'first_response_time': response_time,
        'time_to_resolution': resolution_hours,
        'customer_satisfaction_rating': satisfaction,
    }


def generate_dataset(rows: int):
    return [generate_row(i + 1) for i in range(rows)]


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic customer support ticket data.')
    parser.add_argument('--output', type=str, default='data/sample_tickets.csv', help='Output CSV path.')
    parser.add_argument('--rows', type=int, default=2000, help='Number of rows to generate.')
    args = parser.parse_args()

    data = generate_dataset(args.rows)
    df = pd.DataFrame(data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f'Generated {args.rows} sample rows at {output_path}')


if __name__ == '__main__':
    main()
