FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set env variables
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local'\
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.7.1

RUN pip install "poetry==$POETRY_VERSION"

COPY ["poetry.lock", "pyproject.toml", "./"]

RUN poetry install --only=main --no-root

COPY starter.py . 

RUN mkdir /data
RUN mkdir -p /app/output

ENTRYPOINT [ "python", "starter.py", "--year=2023", "--month=5" ]