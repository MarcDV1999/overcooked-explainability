# Look if the user has introduced all the parameters
if test "$#" -ne 2; then
    cat <<EOF
Usage: $0 ID_AGENT

ID_AGENT:       ID of the agent (To easely save it)
LAYOUT_NAME:    LAYOUT_NAME where the agent will play (simple, unident_s, random0, random1 or random3)

Example: $0 _simple simple
EOF
# If we have all the parameters, we can start the training
else
    echo "Start Testing the Agents ego$1 and alt$1"
    python3 ../PantheonRL/tester.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"'$2'"}' --ego-load ../PantheonRL/models/ego$1 --alt-load ../PantheonRL/models/alt$1
fi


