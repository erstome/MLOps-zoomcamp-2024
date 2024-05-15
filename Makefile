PROJECT_NAME = mlops_zoomcamp

# Configure poetry to create the .venv folder locally
env_conf:
	poetry config virtualenvs.in-project true

# Configurations of the jupyter kernel
jupyter_kernel:
	poetry run python -m ipykernel install --user --name=$(PROJECT_NAME)
