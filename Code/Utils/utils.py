# In this file we have useful tools

# Converts the action (int) in a char
def action_num_to_char(action_num):
    if action_num == 0:
        return "↑"
    elif action_num == 1:
        return '↓'
    elif action_num == 2:
        return '→'
    elif action_num == 3:
        return '←'
    elif action_num == 4:
        return 's'
    elif action_num == 5:
        return 'x'