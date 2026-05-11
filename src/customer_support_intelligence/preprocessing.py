import re
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return text.strip()


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(text) for text in X]


def build_preprocessor(text_column: str, categorical_columns: List[str], numeric_columns: List[str]):
    text_pipeline = Pipeline([
        ('cleaner', TextCleaner()),
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ])

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_pipeline, text_column),
            ('cat', categorical_pipeline, categorical_columns),
            ('num', numeric_pipeline, numeric_columns),
        ],
        remainder='drop',
        sparse_threshold=0,
    )
    return preprocessor


def get_feature_columns():
    text_column = 'ticket_text'
    categorical_columns = ['ticket_channel', 'product_purchased', 'customer_gender']
    numeric_columns = ['customer_age', 'first_response_time']
    return text_column, categorical_columns, numeric_columns
