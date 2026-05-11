import argparse
from pathlib import Path
from src.customer_support_intelligence.inference import TicketInference


def parse_args():
    parser = argparse.ArgumentParser(description='Run a sample ticket prediction.')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory containing saved models.')
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    predictor = TicketInference(model_dir)

    sample = {
        'ticket_subject': 'Unable to access billing page',
        'ticket_description': 'Every time I try to update my payment method the website throws an error.',
        'ticket_channel': 'Email',
        'product_purchased': 'Cloud CRM',
        'customer_age': 34,
        'customer_gender': 'Female',
        'date_of_purchase': '2025-08-15',
        'first_response_time': 1.4,
    }

    result = predictor.predict(sample)
    print('Prediction result:')
    print(result)


if __name__ == '__main__':
    main()
