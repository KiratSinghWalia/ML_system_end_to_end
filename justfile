set dotenv-load
set positional-arguments

KERAS_BACKEND := env("KERAS_BACKEND", "tensorflow")
MLFLOW_TRACKING_URI := env("MLFLOW_TRACKING_URI", "http://localhost:5000")
ENDPOINT_NAME := env("ENDPOINT_NAME", "penguins")




[group('setup')]
@dependencies:
    uv_version=$(uv --version) && \
        just_version=$(just --version) && \
        docker_version=$(docker --version | awk '{print $3}' | sed 's/,//') && \
        jq_version=$(jq --version | awk -F'-' '{print $2}') && \
    echo "uv: $uv_version" && \
    echo "just: $just_version" && \
    echo "docker: $docker_version" && \
    echo "jq: $jq_version"


[group('setup')]
@env:
    if [ ! -f .env ]; then echo "KERAS_BACKEND={{KERAS_BACKEND}}\nMLFLOW_TRACKING_URI={{MLFLOW_TRACKING_URI}}" >> .env; fi
    cat .env
    export $(cat .env | xargs)


[group('setup')]
@mlflow:
    uv run -- mlflow server --host 127.0.0.1 --port 5000 

[group('training')]
@train:
    PYTHONPATH=src uv run python -m pipelines.training --with retry run
@cards:
    PYTHONPATH=src uv run src/pipelines/training.py card server

[group('inference')]
@inference-test:
    PYTHONPATH=src uv run pytest


[group('serving')]
@serve:
    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 uv run -- mlflow models serve \
        -m models:/penguins/$(curl -s -X GET "http://127.0.0.1:5000/api/2.0/mlflow/registered-models/get-latest-versions" \
        -H "Content-Type: application/json" -d '{"name": "penguins"}' \
        | jq -r '.model_versions[0].version') -h 0.0.0.0 -p 8080 --no-conda

@sqlite:
    uv run -- sqlite3 -noheader data/penguins.db "SELECT '• Samples: ' || COUNT(*) || char(10) || '• Labeled: ' || SUM(target IS NOT NULL) || char(10) || '• Unlabeled: ' || SUM(target IS NULL) FROM data;"
