# Overcooked Explainability

## Training Folder

Per a entrenar els models executem les següents comandes

Generem el entorn i instal·lem tot el necessary

```
cd OvercookedCode

docker build -t overcooked_img_bsc .   
```

Entrenem els agents

```bash
docker run -it overcooked_img_bsc  
```

al executar aquesta comanda, estem executant el següent

```bash
./experiments/all_experiments.sh
```

Per tant, estem entrenant tots els agents que especifiquem al arxiu `all_experiments.sh`

## Resultats

Tots els resultats haurien d'estar a `human_aware_rl/human_aware_rl/data/`





---



# Playing with trained agents

```bash
cd human_aware_rl/human_aware_rl

python3 overcooked_interactive.py -t bc -r simple_bc_test_seed4
```

⚠️ Important: Si fas click fora el terminal al executar el codi, ja no podràs seguir jugant.

⚠️ Important 2: En el meu cas (mac), he hagut de desactivar el firewall a configuracio/privacitat i seguretat.







