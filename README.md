# Overcooked Explainability

Overcooked Explainability



# Jugar amb Own trained Agents

Per a poder jugar amb el nostre propi agent entrenat ( cal pujar el model en format pickle) amb l'interfície gràfica.

[Github per pujar el model](https://github.com/HumanCompatibleAI/overcooked-demo)

# Entrenar en nostre propi agent

- [Github per entrenar el nostre propi agent](https://github.com/HumanCompatibleAI/human_aware_rl)
- [Notebook NeurIPS](https://www.notion.so/Training-Overcooked-931ff3dc8feb484896409b3ff7e07c47#9b904261165445ccbc2b84d17edaa98a)

En el README del github s'explica que després de instal·lar tot el necessari, es pot executar la comanda següent per a entrenar a un agent. (En principi s'instal·la tot bé, pero al executar els tests, fallen els del PPO. Per tant, aquesta linia no acaba funcionant.)

```bash
python3 ppo_rllib_client.py with seeds="[1, 2, 3]" lr=1e-3 layout_name=cramped_room num_training_iters=5 num_gpus=0 experiment_name="my_agent"
```

Per tant, primer instal·lem tot el que ens demana:

```bash
git clone --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git

conda create -n harl_rllib python=3.7
conda activate harl_rllib

cd human_aware_rl
./install.sh

pip3 install tensorflow==2.0.2

python3 -c "from ray import rllib"

// Fallan els tests del PPO
./run_tests.sh

export RUN_ENV=local

cd human_aware_rl/ppo
```

I finalment executem (Falla)

```bash
python3 ppo_rllib_client.py with seeds="[1, 2, 3]" lr=1e-3 layout_name=cramped_room num_training_iters=5 num_gpus=0 experiment_name="my_agent"
```

També he provat de instal·lar/executar una branca en concret, la branca [neurips2019](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019). 

```bash
git clone --single-branch --branch neurips2019 --recursive https://github.com/HumanCompatibleAI/human_aware_rl.git

conda create -n harl python=3.7

conda activate harl

cd human_aware_rl
./install.sh

pip3 install tensorflow==1.13.1
conda install mpi4py

python3 run_tests.py
```

Pero tampoc funciona

