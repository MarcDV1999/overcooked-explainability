# Look if the user has introduced all the parameters
if test "$#" -ne 2; then
    cat <<EOF
Usage: $0 ID_AGENT LAYOUT_NAME

ID_AGENT:       ID of the agent (To easely save it) (Example: _simple)
LAYOUT_NAME:    LAYOUT_NAME where the agent will play (simple, unident_s, random0, random1 or random3) (Example: simple)

Example: $0 _simple simple
EOF
# If we have all the parameters, we can start the training
else
    echo "Start Testing the Agents ego$1 and alt$1 in GUI mode"
    python3 ../PantheonRL/overcookedgym/overcooked-flask/app.py --modelpath_p0 ../PantheonRL/models/ego$1 --modelpath_p1 ../PantheonRL/models/alt$1 --layout_name $2
fi


