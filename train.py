import argparse
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from src.customer_support_intelligence.data import load_data, build_ticket_text
from src.customer_support_intelligence.models import (
    build_training_pipelines,
    save_training_artifacts,
    import_optional_transformer,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train support intelligence models.')
    parser.add_argument('--data-path', type=str, default='data/sample_tickets.csv', help='Path to ticket CSV dataset.')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save trained artifacts.')
    parser.add_argument('--test-size', type=float, default=0.2, help='Fraction of data to reserve for testing.')
    parser.add_argument('--use-transformer', action='store_true', help='Train a DistilBERT ticket classifier in addition to classical models.')
    parser.add_argument('--mlflow-experiment', type=str, default='Customer Support Intelligence', help='MLflow experiment name.')
    parser.add_argument('--mlflow-run-name', type=str, default='baseline', help='MLflow run name.')
    parser.add_argument('--mlflow-tracking-uri', type=str, default=None, help='Optional MLflow tracking URI.')
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(dataset_path)
    df = build_ticket_text(df)

    train_df, test_df = train_test_split(df, test_size=args.test_size, stratify=df['ticket_type'], random_state=42)

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params({
            'data_path': str(dataset_path),
            'model_dir': str(model_dir),
            'test_size': args.test_size,
            'use_transformer': args.use_transformer,
        })

        pipelines = build_training_pipelines()

        print('Training ticket type classifier...')
        pipelines['ticket_type'].fit(train_df, train_df['ticket_type'])

        print('Training ticket priority classifier...')
        pipelines['ticket_priority'].fit(train_df, train_df['ticket_priority'])

        print('Training time to resolution regressor...')
        pipelines['resolution_time'].fit(train_df, train_df['time_to_resolution'])

        save_training_artifacts(pipelines, model_dir)

        print('Evaluating models on test set...')
    y_type = test_df['ticket_type']
    y_priority = test_df['ticket_priority']
    y_resolution = test_df['time_to_resolution']

    pred_type = pipelines['ticket_type'].predict(test_df)
    pred_priority = pipelines['ticket_priority'].predict(test_df)
    pred_resolution = pipelines['resolution_time'].predict(test_df)

    type_accuracy = accuracy_score(y_type, pred_type)
    type_f1 = f1_score(y_type, pred_type, average='macro')
    priority_accuracy = accuracy_score(y_priority, pred_priority)
    priority_f1 = f1_score(y_priority, pred_priority, average='macro')
    rmse = mean_squared_error(y_resolution, pred_resolution)
    mae = mean_absolute_error(y_resolution, pred_resolution)
    r2 = r2_score(y_resolution, pred_resolution)

    print('Ticket type classification: accuracy %.3f, f1-macro %.3f' % (type_accuracy, type_f1))
    print('Ticket priority classification: accuracy %.3f, f1-macro %.3f' % (priority_accuracy, priority_f1))
    print('Resolution time regression: RMSE %.3f, MAE %.3f, R2 %.3f' % (
        rmse ** 0.5,
        mae,
        r2))

    mlflow.log_metrics({
        'ticket_type_accuracy': type_accuracy,
        'ticket_type_f1_macro': type_f1,
        'ticket_priority_accuracy': priority_accuracy,
        'ticket_priority_f1_macro': priority_f1,
        'resolution_time_rmse': rmse ** 0.5,
        'resolution_time_mae': mae,
        'resolution_time_r2': r2,
    })

    mlflow.sklearn.log_model(pipelines['ticket_type'], 'ticket_type_model')
    mlflow.sklearn.log_model(pipelines['ticket_priority'], 'ticket_priority_model')
    mlflow.sklearn.log_model(pipelines['resolution_time'], 'resolution_time_model')

    if args.use_transformer:
        transformer = import_optional_transformer()
        if transformer is not None:
            transformer.train_transformer_classifier(train_df, test_df, model_dir)
        else:
            print('Transformer dependencies are not installed. Skipping DistilBERT training.')

    print('Training complete. Artifacts saved to %s' % model_dir)


if __name__ == '__main__':
    main()
