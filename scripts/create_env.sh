#!/bin/bash
set -e

# https://linuxhint.com/pass-named-argument-shell-script/
while [[ "$#" -gt 0 ]]
  do
    case $1 in
      -e|--py_env) ENV_NAME="$2"; shift;;
      -v|--py_version) PY_VER="$2"; shift;;
    esac
    shift
done

echo "Environment name: ${ENV_NAME}"
echo "Python version: ${PY_VER}"

if conda info --envs | grep -q ${ENV_NAME}; then
    echo "Environment ${ENV_NAME} already exists. No action required."; 
    source activate ${ENV_NAME} 
else 
    conda create  --name=${ENV_NAME} python=$PY_VER -y 
    source activate ${ENV_NAME}
    pip3 install -r ./requirements_pip.txt 

    # Install required packages
    conda install --file ./requirements_conda.txt
fi

echo "Environment ${ENV_NAME} succeeded."