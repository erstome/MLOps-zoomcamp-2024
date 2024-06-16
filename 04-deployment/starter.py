import pickle
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction import DictVectorizer
from typing import Tuple, Any

def load_model() -> Tuple[DictVectorizer, Any]:
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model


def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    
    return df

def transform_data(df: pd.DataFrame, dv: DictVectorizer) -> pd.DataFrame:
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    return X_val

def create_ride_id(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df

def create_output(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred
    return df_result

def save_output(df_result: pd.DataFrame, year: int, month: int) -> None:
    output_file = f'./output/df_result_{year:04d}_{month:02d}.parquet'

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def main(year: int = 2023, month: int = 3) -> None:

    df = read_data(f'./data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    dv, model = load_model()

    X_val = transform_data(df, dv)

    y_pred = model.predict(X_val)
    
    df = create_ride_id(df, year, month)
    
    df_result = create_output(df, y_pred)
    
    save_output(df_result, year, month)
    
    print(f'Mean of {year}-{month}: {np.mean(y_pred)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--month', type=int, default=3)
    args = parser.parse_args()
    
    main(year=args.year, month=args.month)
    
    