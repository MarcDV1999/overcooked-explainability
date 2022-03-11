# This script trains a PPO Ego Agent in a particular layout

# Look if the user has introduced all the parameters
if test "$#" -ne 1; then
    cat <<EOF
Usage: $0 ID_AGENT

ID_AGENT:       ID of the agent (To easily save it)

Example: $0 0
EOF
# If we have all the parameters, we can start the training
else
    echo "Start Training of Agents ego$1 and alt$1"
    python3 ../PantheonRL/trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --ego-save ../PantheonRL/models/ego$1 --alt-save ../PantheonRL/models/alt$1
fi
