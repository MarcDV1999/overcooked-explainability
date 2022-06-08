cd Code

# Clone and install PantheonRL
git clone https://github.com/Stanford-ILIAD/PantheonRL.git
cd PantheonRL
pip install -e .

# Install Overcooked environment
git submodule update --init --recursive
pip install -e overcookedgym/human_aware_rl/overcooked_ai

cp ../overcooked.py overcookedgym

cd ../..
pip install -r requirements.txt

#cd Code/PantheonRL
#pip install -e .

#cd ../..
#pip install -r requirements.txt
