#!/bin/bash
set -e 

while [[ "$#" -gt 0 ]]
  do
    case $1 in
      -e|--py_env) ENV_NAME="$2"; shift;;
      -f|--model_config_file) MODEL_CONFIG_FILE="$2"; shift;;
    esac
    shift
done

if [ -z ${ENV_NAME} ]; then
    echo "No valid ENV name...exit."
    exit 1
fi

if [ -z ${MODEL_CONFIG_FILE} ]; then
    echo "Empty model config file...exit."
    exit 1
else
    # https://stackoverflow.com/questions/4175264/how-to-retrieve-absolute-path-given-relative#:~:text=UPD%20Some%20explanations
    export MODEL_CONFIG_FILE=$(cd "$(dirname "${MODEL_CONFIG_FILE}")"; pwd)/$(basename "${MODEL_CONFIG_FILE}")
    echo "Model Configuration File: ${MODEL_CONFIG_FILE}"
fi

# https://gist.github.com/rdonkin/05915bd86475389a94953997f80db9ff
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PROJ_DIR=$(dirname "${SCRIPT_DIR}")

# Create logging folder
if [ ! -d ${PROJ_DIR}/log ]; then
    echo "Creating logging folder: ${PROJ_DIR}/log" && \
    mkdir -p ${PROJ_DIR}/log
else
    echo "Logging folder existed: ${PROJ_DIR}/log"
fi

if conda info --envs | grep -q ${ENV_NAME}; then
    echo "Environment ${ENV_NAME} already exists. No action required."; 
    source activate ${ENV_NAME} 
else 
    echo "Environment ${ENV_NAME} not created...exit"
    exit 1
fi

echo "Current ENV activated: ${CONDA_DEFAULT_ENV}"

export TOKENIZERS_PARALLELISM="false" # To avoid deadlock warning
export SERVICE_PORT=8080
echo "Service at port: "${SERVICE_PORT}
python -m uvicorn src.server:app \
    --port ${SERVICE_PORT} \
    --workers 1 \
    --log-level info \
    --log-config ${PROJ_DIR}/src/logging/logging_config.ini
    # --no-access-log 
    # --reload 