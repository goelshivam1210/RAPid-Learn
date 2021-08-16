import numpy as np
from collections import defaultdict
import time

LOG_STR="tree_log"
PLANK_STR="plank"
STICK_STR="stick"
CT_STR="crafting_table"
TAP_STR="tree_tap"
RUBBER_STR="rubber"
POGO_STR="pogo_stick"
WALL_STR="wall"
AIR_STR="air"
FORWARD_STR="Forward"
LEFT_STR="Left"
RIGHT_STR="Right"
BREAK_STR="Break"
# PLACE_TAP_STR="Place_tree_tap"
EXTRACT_STR="Extract_rubber"
CRAFT_STR="CRAFT_"
SELECT_STR="Select_"
APPROACH_STR="approach"

# for colored print statements
from colorama import Fore, Style, init
init(autoreset=True)

#Util functions
def get_entity_quant(info, item_str):
    try:
        amt = len(info['entities_locs'][item_str])
    except:
        amt = 0
    return amt

def get_world_quant(info, item_str):
    try:
        amt = len(info['items_locs'][item_str])
    except:
        amt = 0
    return amt

def get_inv_quant(info, item_str):
    try:
        amt = info['inv_quant_dict'][item_str]
    except:
        amt = 0
    return amt


## Convert predicate set to function for RL to check success in MDP space
def get_create_success_func_from_predicate(predicate_str):
    create_success_func = None
    if predicate_str.split()[0] == 'holding':
        def create_success_func(init_obs, init_info):
            def success_func(obs, info):
                return info['selected_item'] == predicate_str.split()[1]
            return success_func
    elif predicate_str.split()[0] == 'facing':
        def create_success_func(init_obs, init_info):
            def success_func(obs, info):
                return info['block_in_front']['name'] == predicate_str.split()[1]
            return success_func
    # Numeric fluents - need to support <,<=,==,>=,>,decrease,increase
    elif predicate_str.split()[0] == 'increase':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    world_increase = get_world_quant(info, predicate_str.split()[2]) == get_world_quant(init_info, predicate_str.split()[2]) + int(predicate_str.split()[3])
                    # return len(info['items_locs'][predicate_str.split()[1]]) == len(init_info['items_locs'][predicate_str.split()[1]]) + int(predicate_str.split()[2])
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    inv_increase = get_inv_quant(info, predicate_str.split()[2]) == get_inv_quant(init_info, predicate_str.split()[2]) + int(predicate_str.split()[3])
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] == init_info['inv_quant_dict'][predicate_str.split()[1]] + int(predicate_str.split()[2])
                    return inv_increase
                return success_func
        #For the 'gather log' subgoal, we don't exactly care at the end if we have 2 logs vs 1,
        # Having 1 log and 4 planks is permissible but wouldn't be caught by func above
        elif predicate_str.split()[1] == 'inventory_log':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    desired_increase = int(predicate_str.split()[2])
                    log_increase = get_inv_quant(info, LOG_STR) - get_inv_quant(init_info,LOG_STR)
                    plank_increase = get_inv_quant(info, PLANK_STR) - get_inv_quant(init_info,PLANK_STR)
                    stick_increase = get_inv_quant(info, STICK_STR) - get_inv_quant(init_info,STICK_STR)
                    return log_increase+plank_increase/4+stick_increase/8 >= desired_increase
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] == init_info['inv_quant_dict'][predicate_str.split()[1]] + int(predicate_str.split()[2])
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'decrease':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) == get_world_quant(init_info, predicate_str.split()[2]) - int(predicate_str.split()[3])
                    # return len(info['items_locs'][predicate_str.split()[1]]) == len(init_info['items_locs'][predicate_str.split()[1]]) - int(predicate_str.split()[2])
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    inv_decrease = get_inv_quant(info, predicate_str.split()[2]) == get_inv_quant(init_info, predicate_str.split()[2]) - int(predicate_str.split()[3])
                    # inv_decrease = info['inv_quant_dict'][predicate_str.split()[1]] == init_info['inv_quant_dict'][predicate_str.split()[1]] - int(predicate_str.split()[2])
                    return inv_decrease
                return success_func
    elif predicate_str.split()[0] == 'eq':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) == predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) == predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) == predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] == predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'leq':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) <=  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) <=  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) <=  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] <=  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'l':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) <  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) <  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) <  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] <  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'geq':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) >=  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) >=  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) >=  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] >=  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    elif predicate_str.split()[0] == 'g':
        if predicate_str.split()[1] == 'world':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_world_quant(info, predicate_str.split()[2]) >  predicate_str.split()[2]
                    # return len(info['items_locs'][predicate_str.split()[1]]) >  predicate_str.split()[1]
                return success_func
        elif predicate_str.split()[1] == 'inventory':
            def create_success_func(init_obs, init_info):
                def success_func(obs, info):
                    return get_inv_quant(info, predicate_str.split()[2]) >  predicate_str.split()[2]
                    # inv_increase = info['inv_quant_dict'][predicate_str.split()[1]] >  predicate_str.split()[1]
                    # return inv_increase
                return success_func
    if create_success_func is None:
        print(Fore.RED+"Error, unknown predicate {} supplied".format(predicate_str))
    return create_success_func

def get_create_success_func_from_predicate_set(predicate_list):
    create_success_funcs_set = []
    for predicate_str in predicate_list:
        create_success_funcs_set.append(get_create_success_func_from_predicate(predicate_str))
    def create_success_func(init_obs, init_info):
        def success_func(obs, info):
            # return np.all([fun(init_obs, init_info)(obs, info) for fun in create_success_funcs_set])
            ss = [fun(init_obs, init_info)(obs, info) for fun in create_success_funcs_set]
            return np.all(ss)
        return success_func
    return create_success_func

## Convert prenovelty operator to function for RL to check success in MDP space
def get_create_success_func_from_failed_operator(operator_str):
    # e.g. 'approach {}'.format(LOG_STR) - goal is block_in_front == obj
    if operator_str.split()[0] == APPROACH_STR:
        return get_create_success_func_from_predicate('facing {}'.format(operator_str.split()[1]))
    # to the RL agent approach and pickup should have the same policy though
    if operator_str.split()[0] == 'pickup':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(operator_str.split()[1]), 'decrease world {} 1'.format(operator_str.split()[1])])
    #e.g. break minecraft:log - real goal is inv increase obs, block in front air
    elif operator_str.split()[0] == BREAK_STR:
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(operator_str.split()[1]), 'facing {}'.format(AIR_STR), 'decrease world {} 1'.format(operator_str.split()[1])])
    elif operator_str.split()[0] == 'place':
        return get_create_success_func_from_predicate_set(['decrease inventory {} 1'.format(operator_str.split()[1]), 'facing {}'.format(operator_str.split()[1])])
    # Don't have notion of tapped_log currently
    elif operator_str == EXTRACT_STR:
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(RUBBER_STR)])
    # For all crafting actions goal is to increase
    # Do we need to include notion of decrease as well? - to prevent losing too much?
    #   Do we just make it so the decrease has to be <= the original? Just losing more is not okay
    elif operator_str == 'Craft_plank':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(PLANK_STR)])
    elif operator_str == 'Craft_stick':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(STICK_STR)])
    elif operator_str == 'Craft_tree_tap':
        return get_create_success_func_from_predicate_set(['increase inventory {}} 1'.format(TAP_STR)])
    elif operator_str == 'Craft_pogo_stick':
        return get_create_success_func_from_predicate_set(['increase inventory {} 1'.format(POGO_STR)])
    else:
        print(Fore.RED+"ERROR: cant create effect set from unknown operator {}".format(operator_str))
