# PBT
python3 pbt/pbt.py with fixed_mdp layout_name="simple" EX_NAME="pbt_simple" TOTAL_STEPS_PER_AGENT=8e6 REW_SHAPING_HORIZON=3e6 LR=2e-3 GPU_ID=2 POPULATION_SIZE=3 SEEDS="[8015, 3554,  581, 5608, 4221]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False
python3 pbt/pbt.py with fixed_mdp layout_name="unident_s" EX_NAME="pbt_unident_s" TOTAL_STEPS_PER_AGENT=1.1e7 REW_SHAPING_HORIZON=5e6 LR=8e-4 GPU_ID=3 POPULATION_SIZE=3 SEEDS="[8015, 3554,  581, 5608, 4221]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False # originally 1e-3
python3 pbt/pbt.py with fixed_mdp layout_name="random1" EX_NAME="pbt_random1" TOTAL_STEPS_PER_AGENT=5e6 REW_SHAPING_HORIZON=4e6 LR=8e-4 GPU_ID=1 POPULATION_SIZE=3 SEEDS="[8015, 3554,  581, 5608, 4221]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False
python3 pbt/pbt.py with fixed_mdp layout_name="random0" EX_NAME="pbt_random0" TOTAL_STEPS_PER_AGENT=8e6 REW_SHAPING_HORIZON=7e6 LR=3e-3 GPU_ID=1 POPULATION_SIZE=3 SEEDS="[8015, 3554,  581, 5608, 4221]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False
python3 pbt/pbt.py with fixed_mdp layout_name="random3" EX_NAME="pbt_random3" TOTAL_STEPS_PER_AGENT=6e6 REW_SHAPING_HORIZON=4e6 LR=1e-3 GPU_ID=1 POPULATION_SIZE=3 SEEDS="[8015, 3554,  581, 5608, 4221]" NUM_SELECTION_GAMES=6 VF_COEF=0.5 MINIBATCHES=10 TIMESTAMP_DIR=False



# PPO_BC
python3 ppo/ppo.py with EX_NAME="ppo_bc_train_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_bc_test_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[5e5, 3e6]" TIMESTAMP_DIR=False

python3 ppo/ppo.py with EX_NAME="ppo_bc_train_unident_s" layout_name="unident_s" REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_bc_test_unident_s" layout_name="unident_s" REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=False

python3 ppo/ppo.py with EX_NAME="ppo_bc_train_random1" layout_name="random1" REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_bc_test_random1" layout_name="random1" REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=False

python3 ppo/ppo.py with EX_NAME="ppo_bc_train_random0" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_bc_test_random0" layout_name="random0" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=9e6 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=2 SELF_PLAY_HORIZON=None TIMESTAMP_DIR=False

python3 ppo/ppo.py with EX_NAME="ppo_bc_train_random3" layout_name="random3" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_train" SEEDS="[9456, 1887, 5578, 5987,  516]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_bc_test_random3" layout_name="random3" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=3 OTHER_AGENT_TYPE="bc_test" SEEDS="[2888, 7424, 7360, 4467,  184]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=False

# PPO_HM
python3 ppo/ppo.py with EX_NAME="ppo_hm_simple" layout_name="simple" REW_SHAPING_HORIZON=1e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=1e-3 GPU_ID=0 OTHER_AGENT_TYPE="hm" HM_PARAMS="[True, 1.75, True, 1.7]" SEEDS="[8355, 5748, 1352, 3325, 8611]" VF_COEF=0.5 MINIBATCHES=10 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e2, 1e4]" TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_hm_unident_s" layout_name="unident_s" REW_SHAPING_HORIZON=6e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=1e-3 GPU_ID=1 OTHER_AGENT_TYPE="hm" HM_PARAMS="[True, 1.3, True, 1.1]" SEEDS="[8355, 5748, 1352, 3325, 8611]" VF_COEF=0.5 MINIBATCHES=12 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 7e6]" TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_hm_random1" layout_name="random1" REW_SHAPING_HORIZON=5e6 PPO_RUN_TOT_TIMESTEPS=1.6e7 LR=1e-3 GPU_ID=2 OTHER_AGENT_TYPE="hm" HM_PARAMS="[True, 2, True, 1.8]" SEEDS="[8355, 5748, 1352, 3325, 8611]" VF_COEF=0.5 MINIBATCHES=15 LR_ANNEALING=1.5 SELF_PLAY_HORIZON="[2e6, 6e6]" TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_hm_random3" layout_name="random3" REW_SHAPING_HORIZON=4e6 PPO_RUN_TOT_TIMESTEPS=1.2e7 LR=1.5e-3 GPU_ID=0 OTHER_AGENT_TYPE="hm" HM_PARAMS="[True, 2.2, True, 2]" SEEDS="[8355, 5748, 1352, 3325, 8611]" VF_COEF=0.1 MINIBATCHES=15 LR_ANNEALING=3 SELF_PLAY_HORIZON="[1e6, 4e6]" TIMESTAMP_DIR=False

# PPO_SP
python3 ppo/ppo.py with EX_NAME="ppo_sp_simple" layout_name="simple" REW_SHAPING_HORIZON=2.5e6 LR=1e-3 PPO_RUN_TOT_TIMESTEPS=6e6 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=1 TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_sp_unident_s" layout_name="unident_s" REW_SHAPING_HORIZON=2.5e6 PPO_RUN_TOT_TIMESTEPS=7e6 LR=1e-3 GPU_ID=3 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_sp_random1" layout_name="random1" REW_SHAPING_HORIZON=3.5e6 PPO_RUN_TOT_TIMESTEPS=1e7 LR=6e-4 GPU_ID=0 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_sp_random0" layout_name="random0" REW_SHAPING_HORIZON=2.5e6 PPO_RUN_TOT_TIMESTEPS=7.5e6 LR=8e-4 GPU_ID=2 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False
python3 ppo/ppo.py with EX_NAME="ppo_sp_random3" layout_name="random3" REW_SHAPING_HORIZON=2.5e6 PPO_RUN_TOT_TIMESTEPS=8e6 LR=8e-4 GPU_ID=3 OTHER_AGENT_TYPE="sp" SEEDS="[2229, 7649, 7225, 9807,  386]" VF_COEF=0.5 TIMESTAMP_DIR=False