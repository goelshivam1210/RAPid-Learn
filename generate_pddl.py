import numpy as np
import os
import re
import sys
import csv
import math
import copy
import argparse
import subprocess

def generate_prob_pddl(pddl_dir,env):
    generate_problem_pddl(pddl_dir,env)

def generate_domain_pddl(pddl_dir, env, learned_operator):
    generate_domain(pddl_dir, env, learned_operator)

def generate_domain(pddl_dir, env, learned_operator):
    pass

def generate_problem_pddl(pddl_dir,env, filename: str = "problem"):

    filename = pddl_dir + os.sep + filename + ".pddl"

    hder = _generate_header_prob()
    objs = _generate_objects(env)
    goals = _generate_goals(env)
    ints = _generate_init(env)

    pddl = "\n".join([hder, objs, ints, goals, ")\n"])
    with open(filename, "w", encoding="utf-8") as f:
        f.write(str(pddl))
        print(f"Problem PDDL written to {filename}.")


def _generate_header_prob():
    return f"(define\n\t(problem adeGeneratedProblem)" + "\n\t" + f"(:domain adeGeneratedDomain)"

def _generate_objects(env):
    objects_list = []
    for i in range(env.map_size):
        for j in range(1,env.map_size):
            objects_list.append("\t\tloc_"+str(i)+"_"+str(j)+" - world")
    objects_list.append("\t\tnorth - direction")
    objects_list.append("\t\tsouth - direction")
    objects_list.append("\t\teast - direction")
    objects_list.append("\t\twest - direction")
    objects_list.append("\t\tself - actor")
    objects_list.append("\t\tnothing - material")
    for item in env.items:
        objects_list.append("\t\t"+item+" - material")
    objs = "\n".join(objects_list)
    return "\n".join(["\t(:objects", objs, "\t)"])

def _generate_init(env):
    init_list = []
    for i in range(1,env.map_size):
        for j in range(1, env.map_size):
            init_list.append("\t\t(adjacent loc_"+str(i)+"_"+str(j)+" loc_"+str(i-1)+"_"+str(j)+ " north)")
    for i in range(env.map_size-1):
        for j in range(1, env.map_size):
            init_list.append("\t\t(adjacent loc_"+str(i)+"_"+str(j)+" loc_"+str(i+1)+"_"+str(j)+ " south)")
    for i in range(env.map_size):
        for j in range(1,env.map_size-1):
            init_list.append("\t\t(adjacent loc_"+str(i)+"_"+str(j+1)+" loc_"+str(i)+"_"+str(j)+ " west)")
    for i in range(env.map_size):
        for j in range(1,env.map_size-1):
            init_list.append("\t\t(adjacent loc_"+str(i)+"_"+str(j)+" loc_"+str(i)+"_"+str(j+1)+ " east)")
    init_list.append("\t\t(opposite north south)")
    init_list.append("\t\t(opposite south north)")
    init_list.append("\t\t(opposite east west)")
    init_list.append("\t\t(opposite west east)")
    init_list.append("\t\t(clockwise north east)")
    init_list.append("\t\t(clockwise south west)")
    init_list.append("\t\t(clockwise east south)")
    init_list.append("\t\t(clockwise west north)")

    for i in range(env.map_size):
        init_list.append("\t\t(isBlock wall loc_"+str(i)+"_"+str(env.map_size-1)+")")
        init_list.append("\t\t(at wall loc_"+str(i)+"_"+str(env.map_size-1)+")")
    for i in range(1,env.map_size-1):
        init_list.append("\t\t(isBlock wall loc_"+str(env.map_size-1)+"_"+str(i)+")")        
        init_list.append("\t\t(at wall loc_"+str(env.map_size-1)+"_"+str(i)+")")        
    for i in range(1,env.map_size-1):
        init_list.append("\t\t(isBlock wall loc_0_"+str(i)+")")
        init_list.append("\t\t(at wall loc_0_"+str(i)+")")

    for item in env.items:
        init_list.append("\t\t(= (inventory "+str(item)+") "+str(env.inventory_items_quantity[item])+")")
    init_list.append("\t\t(= ( inventory nothing) 1)")

    for i in range(1,env.map_size-1):
        for j in range(1,env.map_size-1):
            if env.map[i][j] == 0:
                init_list.append("\t\t(isBlock air loc_"+str(i)+"_"+str(j)+")")
                init_list.append("\t\t(at air loc_"+str(i)+"_"+str(j)+")")
            else:
                init_list.append("\t\t(isBlock "+ list(env.items_id.keys())[list(env.items_id.values()).index(env.map[i][j])]+" loc_"+str(i)+"_"+str(j)+")")
                init_list.append("\t\t(at "+ list(env.items_id.keys())[list(env.items_id.values()).index(env.map[i][j])]+" loc_"+str(i)+"_"+str(j)+")")

    for item in env.unbreakable_items:
        init_list.append("\t\t(unbreakable "+item+")")
    init_list.append("\t\t(unbreakable tree_tap)")

    init_list.append("\t\t(tapper tree_tap)")
    init_list.append("\t\t(tapout rubber)")
    init_list.append('\t\t(craftsticksouta stick)')
    init_list.append('\t\t(permeable air)')
    init_list.append('\t\t(craftpogostickinc rubber)')
    init_list.append('\t\t(craftplanksina tree_log)')
    init_list.append('\t\t(craftsticksina plank)')
    init_list.append('\t\t(crafttreetapinb plank)')
    init_list.append('\t\t(crafttreetapina stick)')
    init_list.append('\t\t(crafttreetapouta tree_tap)')
    init_list.append('\t\t(craftpogostickouta pogo_stick)')
    init_list.append('\t\t(craftpogostickinb plank)')
    init_list.append('\t\t(craftpogostickina stick)')
    init_list.append('\t\t(holding nothing)')
    init_list.append('\t\t(crafter crafting_table)')
    init_list.append('\t\t(craftplanksouta plank)')
    init_list.append('\t\t(tappable tree_log)')

    init_list.append("\t\t(at self loc_"+str(env.agent_location[0])+"_"+str(env.agent_location[1])+")")
    init_list.append("\t\t(orientation self "+str(env.agent_facing_str).lower()+")")

    ints = "\n".join(init_list)
    return "\n".join(["\t(:init", ints, "\t)"])

# list(my_dict.keys())[list(my_dict.values()).index(112)]

def _generate_goals(env):
    goal_list = []
    for item in env.goal_item_to_craft:
        goal_list.append(item)
    goal = "".join(goal_list)
    print("goal:", goal)
    # goal = goal_list
    return "".join(["(:goal (>= (inventory ",goal,") 1))"])
 
    