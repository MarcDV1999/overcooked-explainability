#!/bin/sh
# Es fa tota la instalacio i s'executa el run_tests.py
# Si es treballa en MAC pot ser que el run_tests.py dongui error.
# Per a solucionar-lo mirar el README
git clone --single-branch --branch neurips2019 --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git

cd human_aware_rl
conda run -n harl --no-capture-output bash install.sh

conda install -n harl numpy -y
conda install -n harl  pandas -y

pip install tensorflow-gpu==1.13.1
conda install -n harl mpi4py -y
pip install smmap

cd human_aware_rl
#conda run -n harl --no-capture-output python run_tests.py
