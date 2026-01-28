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