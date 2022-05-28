from VNF_Placement import VNF_Placement
from Functions import generate_seeds, read_list_from_file, simple_plot, multi_plot
import numpy as np
import sys
import random

NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 100
NUM_SERVICES = 1
NUM_GAMES = 10000

# generate_seeds(50000)
SEEDS = read_list_from_file("inputs/", "SEEDS_100.txt", "int")

vnf_plc_obj = VNF_Placement(NUM_NODES=NUM_NODES, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS, NUM_GAMES=NUM_GAMES, SEEDS=SEEDS)
# vnf_plc_obj.wf_alloc()
# vnf_plc_obj.ddql_alloc_train()
# vnf_plc_obj.ddql_alloc_eval()
# vnf_plc_obj.rnd_alloc()
# vnf_plc_obj.cm_alloc()
# vnf_plc_obj.dm_alloc()
# vnf_plc_obj.lb_alloc()

# dir = "results/" + vnf_plc_obj.FILE_NAME + "_v3" + "/"
dir = "results/" + vnf_plc_obj.FILE_NAME + "/"
C = ["C1", "C2", "C3", "C4", "C5"]
# L = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
L = ["5e-3", "5e-4", "5e-5", "5e-6", "5e-7"]
def generate_cost_plot_for_different_methods():
    L = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    type = "float"
    avg_win = 200
    lloc = (1, 0.6)  # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    y1 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_ml_avg_ofs" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_wf_avg_ofs" + ".txt", type, round_num=2)
    #y3 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_rnd_avg_ofs" + ".txt", type, round_num=2)
    #y4 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_cm_avg_ofs" + ".txt", type, round_num=2)
    y5 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_dm_avg_ofs" + ".txt", type, round_num=2)
    filename = "results/" + vnf_plc_obj.FILE_NAME + "/" + vnf_plc_obj.FILE_NAME + "_fig" + "_cost_" + str(avg_win) + '.png'
    multi_plot(range(NUM_GAMES), [y1, y2, y5], filename, avg_win, "Cost per Request", C, L, lloc, False)
def generate_reqs_plot_for_different_methods():
    L = ["DDQL-CCRA", "WF-CCRA", "R-CCRA", "CM-CCRA", "DM-CCRA"]
    type = "int"
    avg_win = 200
    lloc = (1, 0.4)  # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    y1 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_ml_reqs" + ".txt", type, round_num=2)
    y2 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_wf_reqs" + ".txt", type, round_num=2)
    # y3 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_rnd_reqs" + ".txt", type, round_num=2)
    # y4 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_cm_reqs" + ".txt", type, round_num=2)
    y5 = read_list_from_file(dir, vnf_plc_obj.FILE_NAME + "_dm_reqs" + ".txt", type, round_num=2)
    filename = "results/" + vnf_plc_obj.FILE_NAME + "/" + vnf_plc_obj.FILE_NAME + "_fig" + "_nar_" + str(avg_win) + '.png'
    multi_plot(range(NUM_GAMES), [y1, y2, y5], filename, avg_win, "Num Active Requests", C, L, lloc, False)
def generate_cost_plot_for_different_eps_decs():
    type = "float"
    avg_win = 100
    lloc = (1, 0.6)  # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    dir_suffix = "/different_eps_decs/"
    L = ["5e-5", "5e-6", "5e-7"]
    y3 = read_list_from_file(dir + dir_suffix, vnf_plc_obj.FILE_NAME + "_ml_5e-5_avg_ofs" + ".txt", type, round_num=2)
    y4 = read_list_from_file(dir + dir_suffix, vnf_plc_obj.FILE_NAME + "_ml_5e-6_avg_ofs" + ".txt", type, round_num=2)
    #y5 = read_list_from_file(dir + dir_suffix, vnf_plc_obj.FILE_NAME + "_ml_5e-7_avg_ofs" + ".txt", type, round_num=2)
    filename = "results/" + vnf_plc_obj.FILE_NAME + "/" + vnf_plc_obj.FILE_NAME + "_fig" + "_diff_eps_decs_cost_" + str(avg_win) + '.png'
    multi_plot(range(NUM_GAMES), [y3, y4, y5], filename, avg_win, "Cost per Request", C, L, lloc, False)
def generate_reqs_plot_for_different_eps_decs():
    type = "float"
    avg_win = 100
    lloc = (1, 0.6)  # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    dir_suffix = "/different_eps_decs/"
    L = ["5e-5", "5e-6", "5e-7"]
    y3 = read_list_from_file(dir + dir_suffix, vnf_plc_obj.FILE_NAME + "_ml_5e-5_reqs" + ".txt", type, round_num=2)
    y4 = read_list_from_file(dir + dir_suffix, vnf_plc_obj.FILE_NAME + "_ml_5e-6_reqs" + ".txt", type, round_num=2)
    y5 = read_list_from_file(dir + dir_suffix, vnf_plc_obj.FILE_NAME + "_ml_5e-7_reqs" + ".txt", type, round_num=2)
    filename = "results/" + vnf_plc_obj.FILE_NAME + "/" + vnf_plc_obj.FILE_NAME + "_fig" + "_diff_eps_decs_reqs_" + str(avg_win) + '.png'
    multi_plot(range(NUM_GAMES), [y3, y4, y5], filename, avg_win, "Number of Active Requests", C, L, lloc, False)

generate_cost_plot_for_different_methods()
generate_reqs_plot_for_different_methods()
# generate_cost_plot_for_different_eps_decs()
# generate_reqs_plot_for_different_eps_decs()