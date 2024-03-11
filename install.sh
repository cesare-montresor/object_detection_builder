#! /bin/bash
#PATH=$PATH:/home/cesare/anaconda3/condabin/

ENV_NAME=odb
PYTHON_VERSION="3.9" #3.9 or higher

conda env remove -y -n $ENV_NAME
if conda info --envs | grep -q $ENV_NAME; then echo "Env already exists, skipping"; else conda create -y -n $ENV_NAME python=$PYTHON_VERSION; fi
conda activate $ENV_NAME

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt