from VNF_Placement import VNF_Placement
import numpy as np
import sys
import random

# from Service_Placement import Service_Placement
# from Priority_Assignment import Priority_Assignment

NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 12
NUM_SERVICES = 1
NUM_GAMES = 11


vnf_plc_obj = VNF_Placement(NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS, NUM_GAMES)
vnf_plc_obj.DDQL()




"""
rewards = np.array(rewards)
x = [i + 1 for i in range(len(rewards))]
plot_learning_curve(range(num_games), ml_avg_ofs, epsilons, filename=file_name + '.png')
"""
