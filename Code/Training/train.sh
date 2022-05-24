# This script trains a PPO Ego Agent in a particular layout

# Look if the user has introduced all the parameters
if test "$#" -ne 3; then
    cat <<EOF
Usage: $0 ID_AGENT LAYOUT

ID_AGENT:       ID of the agent (To easily save it)
LAYOUT:         Layout where the agent will be trained
TIME_STEPS:     Total number of timesteps

Example: $0 _simple simple 500000
EOF
# If we have all the parameters, we can start the training
else
    echo "Start Training of Agents ego$1 and alt$1"
    python3 ../PantheonRL/trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"'$2'"}' --ego-save ../PantheonRL/models/ego$1 --alt-save ../PantheonRL/models/alt$1 -t $3
fi
