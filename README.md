# Overcooked Explainability

## Training Folder

Com generar els models pickles

Creem carpeta on anirà tot

```python
mkdir Training
```

### Overcooked_ai

```
cd Training
```

Anem al github del Overcooked i seguim els passos:

→ https://github.com/HumanCompatibleAI/overcooked_ai/tree/master

```python
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai

git clone <https://github.com/HumanCompatibleAI/overcooked_ai.git>
pip3 install -e overcooked_ai/

cd overcooked_ai
python3 testing/overcooked_test.py
```

### Human_aware_rl

```
cd Training
```

Anem a la branca neurips2019 (es la única que va) i seguim els passos:

→ https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019

```python
git clone --single-branch --branch neurips2019 --recursive <https://github.com/HumanCompatibleAI/human_aware_rl.git>

conda create -n harl2 python=3.7
conda activate harl2

cd human_aware_rl
./install.sh

conda install numpy
conda install pandas

pip3 install tensorflow==1.13.1
conda install mpi4py
pip3 install smmap

cd human_aware_rl
python3 run_tests.py
```

En macOS potser surt un error tipo `Python must be installed as a framework.`

Per solucionar-ho:

→ [Telling Matplotlib to use a different backend](https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/)

Anar al fitxer `human_aware_rl/human_aware_rl/imitation/behavioural_cloning.py`

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







