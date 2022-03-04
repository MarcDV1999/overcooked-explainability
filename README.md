<h1 align="center">
  <br>
  <a href="https://aessestudiants.upc.edu"><img src="Images/README/logo-upc.png" alt="UPC Logo" width="500"></a>
	<br>
  <br>
  Explaining Overcooked RL Agent 🧑‍🍳🤖
  <br>
</h1>

In this repo we will explain the behaviour of an agent trained to perform well on overcooked using Reinforcement Learning.



# 🏗 Built with

- [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL)



# ✅ Installation



## 🤖 PantheonRL Installation

It is useful to setup a conda environment with Python 3.7 (virtualenv works too):

```bash
# Optionally create conda environments
conda create -n PantheonRL python=3.7
conda activate PantheonRL
```
In this project, we use a modified version of the PantheonRL repository but if we wanted to use the original code of PantheonRL, we could install it with the following command lines:
```bash
# Clone and install PantheonRL. 
git clone https://github.com/Stanford-ILIAD/PantheonRL.git
cd PantheonRL
pip install -e .

# Finally install Overcooked
git submodule update --init --recursive
pip3 install -e overcookedgym/human_aware_rl/overcooked_ai
```

---



# 🤖 Training our RL Overcooked Agent

In our case we will use an agent trained with a RL technique called [PPO (Proximal Policy Optimization)](). To do so, we can execute the following command line.

```bash
cd Code
# The parameter is the ID of the new agent
bash train.sh
```

Once the training had been finished, we will be able to see the following trained agents in [`models`](Code/PantheonRL/models) folder:

- **Ego Agent:** The ego-agent is considered the main agent in the environment. From the perspective of the ego agent, the environment functions like a regular gym environment.
- **Alt Agents:** The alt-agents are the partner agents that are embedded in the environment. If multiple are listed, the environment randomly samples one of them to be the partner at the start of each episode.

If we want to personalize more your agent, you could use the following command line and add all the configurations that you want:

```bash
cd PantheonRL

python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --ego-save models/ego --alt-save models/alt
```

---



# 🧪 Test our RL Overcooked Agent

We can test our agents with the following command line:

```bash
# The parameter is the ID of the agents
bash test.sh 0
```

Once the testing had been finished, we will be able to see the mean episode reward and other useful information.

If we want to personalize more your agent, you could use the following command line and add all the configurations that you want:

```bash
cd PantheonRL

python3 tester.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --ego-load models/ego --alt-load models/alt
```

Also we can test the agent in a web interface:

```bash
python3 overcookedgym/overcooked-flask/app.py --modelpath_p0 models/ego1 --modelpath_p1 models/alt1 --layout_name simple
```

---



# 🕸 Building Policy Graph

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











---



# 🏛 Repo Structure Overview

- Train/Test
  - [`tester.py`](Code/PantheonRL/tester.py): Code for testing the Agents.
  - [`trainer.py`](Code/PantheonRL/trainer.py): Code for training the Agents.
  - [`bctrainer.py`](Code/PantheonRL/bctrainer.py): Code for training a BC Agent.
  - [`app.py`](Code/PantheonRL/overcookedgym/overcooked-flask/app.py): Executes a Flask App where we can see the agents playing in a GUI.
- Examples
  - [`OvercookedAdaptPartnerInstructions.md`](Code/PantheonRL/overcookedgym/OvercookedAdaptPartnerInstructions.md): Training with terminal examples.
  - [`OvercookedFlaskWebAppInstructions.md`](Code/PantheonRL/overcookedgym/OvercookedFlaskWebAppInstructions.md): Web app examples.
  - [`overcookedtraining.py`](Code/PantheonRL/examples/overcookedtraining.py): Example of how to train an Agent with Python.
  
- Environment
  - [`overcooked_env.py`](Code/PantheonRL/overcookedgym/human_aware_rl/overcooked_ai/overcooked_ai_py/mdp/overcooked_env.py): Overcooked environment.

- Policy Graph Extraction
  - [`arguments_utils.py`](Code/arguments_utils.py): Helps with the argument parsing work.
  - [`get_policy_graph.py`](Code/get_policy_graph.py): Extracts the Policy Graph of an Agent.
  - [`overcooked.py`](Code/PantheonRL/overcookedgym/overcooked.py): Implements the `OvercookedMultiEnv(SimultaneousEnv)` class.  In this class 
  



# 📘 Research Papers

- Carroll, Micah, Rohin Shah, Mark K. Ho, Thomas L. Griffiths, Sanjit A. Seshia, Pieter Abbeel, and Anca Dragan. ["On the utility of learning about humans for human-ai coordination."](https://arxiv.org/abs/1910.05789) NeurIPS 2019.





# 🔬Contributing

1. Fork the project (<https://github.com/yourname/yourproject/fork>)
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -am 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a new Pull Request



# ➕ More Information (Optional)

For more information about the project, see the following document: 

- [Paper]()



# 🙋‍♂️ Authors

  - **Marc Domènech**  - [MarcDV1999](https://github.com/MarcDV1999)



# 🎓 License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/MarcDV1999/Traffic-Signals-Predictor/blob/main/LICENSE) file for details
