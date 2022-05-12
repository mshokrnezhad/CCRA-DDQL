from VNF_Placement import VNF_Placement
from Functions import generate_seeds, read_list_from_file
import numpy as np
import sys
import random

# from Service_Placement import Service_Placement
# from Priority_Assignment import Priority_Assignment


NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 12
NUM_SERVICES = 1
NUM_GAMES = 20
# generate_seeds(5000)
SEEDS = read_list_from_file("inputs/", "SEEDS.txt", "int")

vnf_plc_obj = VNF_Placement(NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS, NUM_GAMES, SEEDS)
vnf_plc_obj.WF()



"""
rewards = np.array(rewards)
x = [i + 1 for i in range(len(rewards))]
plot_learning_curve(range(num_games), ml_avg_ofs, epsilons, filename=file_name + '.png')
"""
