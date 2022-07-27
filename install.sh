# Init and update all the submodules
git submodule update --init --recursive

# Install PantheonRL
cd Code/PantheonRL
pip install -e .

# Install Overcooked 
git submodule update --init --recursive
pip install -e overcookedgym/human_aware_rl/overcooked_ai

cp ../overcooked.py overcookedgym

cd ../..
pip install -r requirements.txt
