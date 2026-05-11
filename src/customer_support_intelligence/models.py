import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

from .preprocessing import build_preprocessor, get_feature_columns


def build_training_pipelines():
    text_column, categorical_columns, numeric_columns = get_feature_columns()
    preprocessor = build_preprocessor(text_column, categorical_columns, numeric_columns)

    ticket_type_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42)),
    ])

    priority_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=120, random_state=42, class_weight='balanced')),
    ])

    resolution_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(random_state=42, n_estimators=120, learning_rate=0.1)),
    ])

    return {
        'ticket_type': ticket_type_pipeline,
        'ticket_priority': priority_pipeline,
        'resolution_time': resolution_pipeline,
    }


def save_training_artifacts(pipelines: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, pipeline in pipelines.items():
        filename = output_dir / f'{name}.joblib'
        joblib.dump(pipeline, filename)


def import_optional_transformer():
    try:
        import importlib
        transformer_module = importlib.import_module('src.customer_support_intelligence.transformer_model')
        return transformer_module
    except ImportError:
        return None
