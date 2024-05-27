PROJECT_NAME = mlops_zoomcamp

# Configure poetry to create the .venv folder locally
env_conf:
	poetry config virtualenvs.in-project true

# Configurations of the jupyter kernel
jupyter_kernel:
	poetry run python -m ipykernel install --user --name=$(PROJECT_NAME)

# Start MLFlow server
mlflow_server:
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts
