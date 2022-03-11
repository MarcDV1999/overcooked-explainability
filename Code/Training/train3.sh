# This script trains a PPO Ego Agent in multiple layouts and seeds

#layouts=(corridor five_by_five random0 random1 random2 random3 scenario1_s scenario2 scenario2_s scenario3 scenario4 schelling schelling_s simple small_corridor unident unident_s)
layouts=(simple unident_s random1 random0 random3)
seed_ini=1
seed_end=10
seed_increment=1

actual_iter=1
num_layouts=${#layouts[@]}
num_iters=$((num_layouts*((seed_end-seed_ini)/seed_increment)))

# Look if the user has introduced all the parameters
if test "$#" -ne 1; then
    cat <<EOF
Usage: $0 ID_AGENT

ID_AGENT:       ID of the agent (To easely save it)

Example: $0 0
EOF

# If we have all the parameters, we can start the training
else
    first_layout=${layouts[0]}
    progress=$((100*actual_iter/num_iters))
    echo -e "\n------------------------------------------- Process $progress%"
    echo "------------------------------------------- Training ego$1 to play in --> layout: $first_layout - seed: 0"
    python3 ../PantheonRL/trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"'$first_layout'"}' --ego-save ../PantheonRL/models/ego$1 -s 1
    echo -e "\n"

    # shellcheck disable=SC2068
    for ((seed=seed_ini; seed<seed_end; seed=seed+seed_increment))
    do
      for l in ${layouts[@]}
      do
        progress=$((100*actual_iter/num_iters))
        echo -e "\n------------------------------------------- Process $progress%"
        echo "------------------------------------------- Training ego$1 to play in --> layout: $l - seed: $seed"
        python3 ../PantheonRL/trainer.py OvercookedMultiEnv-v0 LOAD PPO --ego-config '{"type":"PPO", "location":"'../PantheonRL/models/ego$1'"}' --ego-save ../PantheonRL/models/ego$1 --env-config '{"layout_name":"'$l'"}' --seed $seed
        echo -e "\n"
        actual_iter=$((actual_iter+1))
      done
    done

fi
