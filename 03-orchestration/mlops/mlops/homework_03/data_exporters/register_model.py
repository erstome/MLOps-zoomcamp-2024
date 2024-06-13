if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import pickle
import mlflow
from mlflow import log_artifact


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    # Specify your data exporting logic here
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("homework_03_experiment")

    with open('lin_reg.bin', 'wb') as f:
        pickle.dump(data, f)

    with mlflow.start_run() as run:
    
        mlflow.set_tag("developer", "Emanuel Tomé")
        
        mlflow.log_artifact(local_path="lin_reg.bin",
                            artifact_path="models_pickle")
        
        mlflow.sklearn.log_model(data, artifact_path='models')
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")

        

