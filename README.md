<h1 align="center">
  <br>
  <a href="https://hpai.bsc.es/"><img src="Images/README/logo-upc.png" alt="FIB, UPC Logo" width="500"></a>
  <br></br>
    Explaining Overcooked RL Agent 🧑‍🍳🤖
	</br>
</h1>
<h3 align="center">
  Marc Domènech i Vila, Sergio Álvarez Napagao and Dmitry Gnatyshak
</h3>
<p align="center">
  Artificial Intelligence Research and Development
</p>




Explainable AI can be used to find ways to explain the way of reasoning of some AI algorithms that we see as a black box. This project aims to be the continuation of another[^1] where good results were achieved in a simple environment such as CartPole using a behavior graph extracted from RL-trained agents. The aim of this work is to see if an agent trained with these techniques can perform as well as an agent trained with RL with a more complex coordination environment such as Overcooked.



# 🏗 Built with

---

- [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL)
- [Overcooked AI](https://github.com/HumanCompatibleAI/overcooked_ai)



# ✅ Installation

---

This repo has already all we need. ⚠️ It is **important to use the files included in this repo** because I have had to modify some files from PantheonRL in order to get the agent observation as we wanted. ⚠️.

The only thing you have to install is:

```bash
pip install networkx==2.6.3   
```



# :shallow_pan_of_food: Introduction

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

# 🤖 Training our RL Agent

---

In our case we will use an agent trained with a RL technique called [PPO (Proximal Policy Optimization)](). To do so, we can execute one of the following command lines.

```bash
cd Code/Training

# Train an agent in layout: simple
# The parameter is the ID of the new agent
bash train.sh 0
```

Once the training had been finished, we will be able to see the following trained agents in [`models`](Code/PantheonRL/models) folder:

- **Ego Agent:** The ego-agent is considered the main agent in the environment. From the perspective of the ego agent, the environment functions like a regular gym environment.
- **Alt Agents:** The alt-agents are the partner agents that are embedded in the environment. If multiple are listed, the environment randomly samples one of them to be the partner at the start of each episode.

If we want to personalize more your agent, you could use the following command line and add all the configurations that you want:

```bash
cd PantheonRL

python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --ego-save models/ego0 --alt-save models/alt0
```

In this case we are training our agent to perform well in a particular environment, `simple` in this case. I have developed two more scripts that trains an agent in multiple layouts and seeds. To use it, we can execute the following command line.

```bash
cd Code/Training

# Train an agent in layouts: (simple unident_s random1 random0 random3) 
# and for each layout uses seeds = range(1,10,1)
# The parameter is the ID of the new agent
bash train2.sh 0

# Train an agent in seeds = range(1,10,1)
# and for each seed uses layouts: (simple unident_s random1 random0 random3) 
# The parameter is the ID of the new agent
bash train3.sh 0
```



>## 👍 Trained Models
>
>When cloning the repository you will see that in [`models`](Code/PantheonRL/models) folder, there are already trained models. Here I attach a brief summary of each one.
>
>- ❌ [`ego51`](Code/PantheonRL/models/ego51.zip): Agent trained in `layouts = [simple, unident_s, random1, random0, random3]` and for each layout `seeds = range(1, 10, 1)`.
>- 🔜 [`ego10`](Code/PantheonRL/models/ego10.zip): Agent trained in `seeds = range(1, 10, 1)` and for each seed, `layouts = [simple, unident_s, random1, random0, random3]`.
>- ✅ [`ego_simple`](Code/PantheonRL/models/ego_simple.zip): Agent trained to perform well in `simple` layout.
>

After this experimentation we saw that training an agent with the aim of performing well on multiple layouts, was not a good approach (we have to figure out an explanation). For this reason, we decided to train an agent that performed well in a single layout, in our case the `simple` layout.

# 🧪 Test our RL Agent

---

We can test our agents with the following command line:

```bash
cd Code/Testing
# The first parameter is the ID of the agents
# The second parameter is the layout
bash test.sh 0 simple
```

Once the testing had been finished, we will be able to see the mean episode reward and other useful information.

If we want to personalize more your agent, you could use the following command line and add all the configurations that you want:

```bash
cd PantheonRL

python3 tester.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --ego-load models/ego --alt-load models/alt
```

Also we can test the agent in a web interface:

```bash
cd Code/Testing
# The first parameter is the ID of the agents
# The second parameter is the layout
bash test_GUI.sh 0 simple
```



## Tested models

| Model      | Description | Mean Episode Reward | Test Result | Seeds         | STD    |
| ---------- | ----------- | ------------------- | ----------- | ------------- | ------ |
| ego_simple |             | 388.14              | ❌           | range(1,10,2) | 15.817 |
|            |             |                     |             |               |        |
|            |             |                     |             |               |        |



# 🕸 Building Policy Graph

---

Here we are using a method called Policy Graph. In this method what we are trying to do is to generate a Graph that represents de MDP of the agent. Since our environment is discrete, we won't have to discretize their states. The states is represented as follows:

```json
State{
	Players: ((2, 1) facing (0, -1) holding None, 
						(3, 2) facing (0, 1) holding soup@(3, 2) with state ('onion', 3, 20)),
	Objects: [soup@(2, 0) with state ('onion', 1, 0)], 
	Order list: None
}
```

This is a [`overcooked_ai_py.mdp.overcooked_mdp.OvercookedState`](Code/PantheonRL/overcookedgym/human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py). 

On the other hand, in the original code, they featurize this state into somtehing that has the following shape:

```python
State = [ 0.  1.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. -1. -2.  0.
  0. -2.  1.  0.  0.  0.  1.  0.  1.  1.  0.  1.  0.  0.  0.  0.  0.  1.
 -2.  0.  0.  0.  0.  0.  0.  0.  0. -1.  0.  0.  0.  0.  0.  0.  1.  2.
  1.  0.  0.  0. -1. -1.  3.  2.]
```

This is a `numpy.ndarray` of length 62. Currently, I don't know what it means 😅.

We have a third option that is represent the state like this:

```bash
X Xoø18X X 
O   ↑d  O 
Xd    ↓oX 
X D X S X 
```

This is a matrix where each position corresponds to a cell of the layout.



Currently, I think the first is the better one but I'm not pretty sure. Once we decide which representation is the best, we had to ask ourselves, which predicates we will use to represent the nodes of the graph.

## State Representation

```bash
X X P X X 
O   ←1  O 
X ↓0    X 
X D X S X 
```



## Possible Discretization

Finally we decided to use the first representation of the state.

```json
State{
	Players: ((2, 1) facing (0, -1) holding None, 
						(3, 2) facing (0, 1) holding soup@(3, 2) with state ('onion', 3, 20)),
	Objects: [soup@(2, 0) with state ('onion', 1, 0)], 
	Order list: None
}
```

In order to discretisize the observation, we probably would have to use the following info:

- Objects that holds the agent.
- Direction and Distance to the other objects. (Needed since the PG would have to predict an action given an state)
- Can we know if the oven is full? and how much is the remaining time?



# 🏛 Repo Structure Overview

---

Here we can see a brief summary of the repo structure.

## 🧑🏼‍💻 Code Folder

- [`Training`](Code/Training): Code related with the training of the agents.
  - [`train.sh`](Code/Training/train.sh): Script that trains an Ego and Alt agent using PPO in a particular layout.
  - [`train2.sh`](Code/Training/train2.sh): Script that trains an Ego agent using PPO in multiple layouts and seeds.
  - [`train3.sh`](Code/Training/train3.sh): Script that trains an Ego agent using PPO in multiple seeds and layouts.
  
- [`Testing`](Code/Testing): Code related with the testing of the agents.
  - [`test.sh`](Code/Testing/test.sh): Script that tests a trained Ego agent in a particular layout.
  - [`test_GUI.sh`](Code/Testing/test_GUI.sh): Script that tests a trained Ego agent using GUI.

- [`Explainability`](Code/Explainability): Code related with the agent explainability.
  - [`get_policy_graph.py`](Code/app.py): Extracts the Policy Graph of an Agent.
  - [`PolicyGraph.py`](Code/Explainability/PolicyGraph.py): Class that saves and computes the Policy Graph.

- [`Utils`](Code/Utils): Code related with useul tools.
  - [`arguments_utils.py`](Code/Utils/arguments_utils.py): Code that helps with the argument parsing.
  - [`utils.py`](Code/Utils/utils.py): Code with usefool tools.




## 🦾 PantheonRL Folder

This folder has a lot of files. Here I mention those files that I think are more interesting.

- Train/Test
  - [`tester.py`](Code/PantheonRL/tester.py): Code for testing the Agents.
  - [`trainer.py`](Code/PantheonRL/trainer.py): Code for training an Agents in a particular layout.
  - [`bctrainer.py`](Code/PantheonRL/bctrainer.py): Code for training a BC Agent.
  - [`app.py`](Code/PantheonRL/overcookedgym/overcooked-flask/app.py): Executes a Flask App where we can see the agents playing in a GUI.
- Examples
  - [`OvercookedAdaptPartnerInstructions.md`](Code/PantheonRL/overcookedgym/OvercookedAdaptPartnerInstructions.md): Training with terminal examples.
  - [`OvercookedFlaskWebAppInstructions.md`](Code/PantheonRL/overcookedgym/OvercookedFlaskWebAppInstructions.md): Web app examples.
  - [`overcookedtraining.py`](Code/PantheonRL/examples/overcookedtraining.py): Example of how to train an Agent with Python.
- Environment
  - [`overcooked_env.py`](Code/PantheonRL/overcookedgym/human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_env.py): Overcooked environment.
  - [`overcooked_mdp.py`](Code/PantheonRL/overcookedgym/human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_mdp.py): Overcooked state in class `OvercookedState`. Player state in class `PlayerState`.
- Policy Graph Extraction
  - [`overcooked.py`](Code/PantheonRL/overcookedgym/overcooked.py): Implements the `OvercookedMultiEnv(SimultaneousEnv)` class.  I have modified the `multi_step()` in order to also return the observation with the shape we want.



# 📗 Glossary

---

- **Episode:** It refers to a Game. One game consists in taking 400 actions, this means that an agent will take 400 actions over the game.
- **Epoch:** It refers to a set of Episodes. 

# 📘 Research Papers

---

- [^1]: [A. Climent, D. Gnatyshak, and S. Alvarez-Napagao."Applying and verifying an explainability method based on policy graphs in the context of reinforcement learning" 7 2021.](https://ebooks.iospress.nl/volumearticle/57744)

- Carroll, Micah, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, and Anca Dragan. ["On the utility of learning about humans for human-ai coordination."](https://arxiv.org/abs/1910.05789) NeurIPS 2019.

- [^2]: [Carroll, Micah, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, and Anca Dragan. "On the utility of learning about humans for human-ai coordination." NeurIPS 2019.](https://arxiv.org/abs/1910.05789)

  





# 🔬Contributing

---

1. Fork the project (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -am 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a new Pull Request



# ➕ More Information

---

For more information about the project, see the following document: 

- [Paper/TFG]()



# 🙋‍♂️ Authors

---

### Thesis student

- **Marc Domènech**  - [MarcDV1999](https://github.com/MarcDV1999)

### Thesis supervisor

- **Sergio Álvarez Napagao** - ([salvarez@cs.upc.edu](mailto:salvarez@cs.upc.edu))

### Co-supervisor

- **Dmitry Gnatyshak**





# 🎓 License

---

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
