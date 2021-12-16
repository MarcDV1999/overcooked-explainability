#!/bin/sh
git clone --single-branch --branch neurips2019 --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git

cd human_aware_rl
conda run -n harl --no-capture-output bash install.sh

conda install -n harl numpy -y
conda install -n harl  pandas -y
conda install -n harl  scipy -y
conda install -n harl matplotlib -y
pip install memory-profiler
pip install sacred

conda install -n harl tensorflow-gpu=1.13.1 -y
conda install -n harl mpi4py -y
conda install -n harl gitdb -y

cd human_aware_rl
