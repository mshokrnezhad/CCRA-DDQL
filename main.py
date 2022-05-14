from VNF_Placement import VNF_Placement
from Functions import generate_seeds, read_list_from_file, simple_plot
import numpy as np
import sys
import random

# from Service_Placement import Service_Placement
# from Priority_Assignment import Priority_Assignment


NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 12
NUM_SERVICES = 1
NUM_GAMES = 20000

# generate_seeds(50000)
SEEDS = read_list_from_file("inputs/", "SEEDS.txt", "int")

vnf_plc_obj = VNF_Placement(NUM_NODES=NUM_NODES, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS, NUM_GAMES=NUM_GAMES, SEEDS=SEEDS)
# vnf_plc_obj.ddql_train()
# vnf_plc_obj.ddql_eval()
# vnf_plc_obj.wf()

# dir = "results/" + vnf_plc_obj.FILE_NAME + "_v3" + "/"
dir = "results/" + vnf_plc_obj.FILE_NAME + "/"

type = "float"
ml_avg_ofs = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)
opt_avg_ofs = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_opt_avg_ofs" + ".txt", type, round_num=2)
simple_plot(range(NUM_GAMES), ml_avg_ofs, opt_avg_ofs, filename="results/" + vnf_plc_obj.FILE_NAME + "/" + vnf_plc_obj.FILE_NAME + "_opt_vs_ml_of" + '.png')

"""
type = "int"
ml_nums_act_reqs = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_ml_nums_act_reqs" + ".txt", type, round_num=2)
opt_nums_act_reqs = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_opt_nums_act_reqs" + ".txt", type, round_num=2)
simple_plot(range(NUM_GAMES), ml_nums_act_reqs, opt_nums_act_reqs, filename=dir + vnf_plc_obj.FILE_NAME + "_opt_vs_ml_num_act_reqs" + '.png')
"""