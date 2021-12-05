# Overcooked Explainability

## Training Folder

```
cd OvercookedCode
```

Anem a la branca neurips2019 (es la única que va) i seguim els passos:

→ https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019

```python
conda create -n harl python=3.7
conda activate harl

./install.sh

cd human_aware_rl
python3 run_tests.py
```

En macOS potser surt un error tipo `Python must be installed as a framework.`

Per solucionar-ho([Telling Matplotlib to use a different backend](https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/))

Anar al fitxer `human_aware_rl/human_aware_rl/imitation/behavioural_cloning.py`

```
open human_aware_rl/human_aware_rl/imitation/behavioural_cloning.py   
```

I substituir la linia:

```python
import matplotlib.pyplot as plt
```

Per:

```python
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
```

## Resultats

En principi a `human_aware_rl/human_aware_rl/data/` és troben els models pickle.



---



# Executar pickle a algún lloc

Ara s'hauria de veure com es pot fer servir aquests pickles






