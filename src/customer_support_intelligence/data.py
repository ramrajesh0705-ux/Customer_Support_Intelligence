from pathlib import Path
import re
import pandas as pd

DATA_COLUMNS = [
    'ticket_id',
    'customer_name',
    'customer_email',
    'customer_age',
    'customer_gender',
    'product_purchased',
    'date_of_purchase',
    'ticket_type',
    'ticket_subject',
    'ticket_description',
    'ticket_status',
    'resolution',
    'ticket_priority',
    'ticket_channel',
    'first_response_time',
    'time_to_resolution',
    'customer_satisfaction_rating',
]

COLUMN_ALIASES = {
    'ticket_id': ['ticket id', 'ticketid'],
    'customer_name': ['customer name'],
    'customer_email': ['customer email'],
    'customer_age': ['customer age'],
    'customer_gender': ['customer gender'],
    'product_purchased': ['product purchased'],
    'date_of_purchase': ['date of purchase'],
    'ticket_type': ['ticket type'],
    'ticket_subject': ['ticket subject'],
    'ticket_description': ['ticket description'],
    'ticket_status': ['ticket status'],
    'resolution': ['resolution'],
    'ticket_priority': ['ticket priority'],
    'ticket_channel': ['ticket channel'],
    'first_response_time': ['first response time'],
    'time_to_resolution': ['time to resolution'],
    'customer_satisfaction_rating': ['customer satisfaction rating'],
}


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', column_name.strip().lower()).strip('_')


def _parse_duration_string(value):
    if pd.isna(value):
        return None
    try:
        duration = pd.to_timedelta(value, errors='coerce')
        if not pd.isna(duration):
            return duration.total_seconds() / 3600.0
    except Exception:
        pass
    if isinstance(value, str):
        match = re.search(r'(\d{1,2}:\d{2}:\d{2})$', value.strip())
        if match:
            duration = pd.to_timedelta(match.group(1))
            return duration.total_seconds() / 3600.0
    return None


def _parse_timestamp_to_hours(value):
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        timestamp = pd.to_datetime(value, errors='coerce')
        if pd.isna(timestamp):
            return None
        return timestamp.timestamp() / 3600.0
    except Exception:
        return None


def _convert_to_numeric_hours(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors='coerce')
    result = numeric.copy()

    missing_mask = result.isna() & ~series.isna()
    if missing_mask.any():
        parsed_ts = pd.to_datetime(series[missing_mask], errors='coerce')
        result.loc[missing_mask] = parsed_ts.view('int64') / 1e9 / 3600.0

        still_missing = result.isna() & ~series.isna()
        if still_missing.any():
            result.loc[still_missing] = series[still_missing].apply(_parse_duration_string)

    return result


def _convert_time_to_resolution(series: pd.Series, first_response_series: pd.Series) -> pd.Series:
    result = _convert_to_numeric_hours(series)

    if result.isna().any():
        response_dt = pd.to_datetime(first_response_series, errors='coerce')
        resolution_dt = pd.to_datetime(series, errors='coerce')
        if not resolution_dt.isna().all() and not response_dt.isna().all():
            delta = (resolution_dt - response_dt).dt.total_seconds() / 3600.0
            result = result.fillna(delta)

    if result.isna().any():
        still_missing = result.isna() & ~series.isna()
        if still_missing.any():
            result.loc[still_missing] = series[still_missing].apply(_parse_duration_string)

    return result


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    rename_map = {}
    normalized_aliases = {
        alias: canonical
        for canonical, aliases in COLUMN_ALIASES.items()
        for alias in aliases
    }

    for original_col in df.columns:
        normalized_col = _normalize_column_name(original_col)
        if normalized_col in DATA_COLUMNS:
            rename_map[original_col] = normalized_col
        elif normalized_col in normalized_aliases:
            rename_map[original_col] = normalized_aliases[normalized_col]

    if rename_map:
        df = df.rename(columns=rename_map)

    missing = set(DATA_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')

    df = df.copy()
    raw_first_response = df['first_response_time']
    df['first_response_time'] = _convert_to_numeric_hours(raw_first_response)
    df['time_to_resolution'] = _convert_time_to_resolution(
        df['time_to_resolution'],
        raw_first_response,
    )

    df = df.dropna(subset=['ticket_type', 'ticket_priority', 'time_to_resolution']).reset_index(drop=True)
    return df


def build_ticket_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['ticket_text'] = (
        df['ticket_subject'].fillna('') + ' ' + df['ticket_description'].fillna('')
    )
    return df
