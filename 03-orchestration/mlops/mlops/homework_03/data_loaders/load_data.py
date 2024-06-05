import io
import pandas as pd
import requests
if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data_from_api(*args, **kwargs):
    """
    Template for loading data from API
    """
    #url = 'https://github.com/erstome/MLOps-zoomcamp-2024/blob/main/data/yellow_tripdata_2023-03.parquet'
    url = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet'
    response = requests.get(url)

    assert response.status_code == 200, "The download was not successful."

    df = pd.read_parquet(io.BytesIO(response.content))

    return df


# @test
# def test_output(output, *args) -> None:
#     """
#     Template code for testing the output of the block.
#     """
#     assert output is not None, 'The output is undefined'