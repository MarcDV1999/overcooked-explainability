#!/bin/sh
# Es fa tota la instalacio i s'executa el run_tests.py
# Si es treballa en MAC pot ser que el run_tests.py dongui error.
# Per a solucionar-lo mirar el README
git clone --single-branch --branch neurips2019 --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git

cd human_aware_rl
./install.sh

conda install numpy -y
conda install pandas -y

pip3 install tensorflow==1.13.1
conda install mpi4py -y
pip3 install smmap

cd human_aware_rl
python3 run_tests.py
