# from docplex.mp.model import Model
import numpy as np
import matplotlib.pyplot as plt
import os.path

rnd = np.random


def specify_requests_entry_nodes(FIRST_TIER_NODES, REQUESTS, seed=0):
    rnd.seed(seed)
    return np.array([rnd.choice(FIRST_TIER_NODES) for i in REQUESTS])


def assign_requests_to_services(SERVICES, REQUESTS, seed=1):
    rnd.seed(seed)
    return np.array([rnd.choice(SERVICES) for i in REQUESTS])


def calculate_input_shape(NUM_NODES, NUM_REQUESTS, NUM_PRIORITY_LEVELS, switch):
    counter = 0

    if switch == "srv_plc":
        counter = (2 * NUM_REQUESTS) + (5 * NUM_NODES) + ((2 + NUM_PRIORITY_LEVELS) * (NUM_NODES ** 2))
    """
    if switch == "srv_plc":
        counter = (2 * NUM_REQUESTS) + (5 * NUM_NODES) + ((2) * (NUM_NODES ** 2))
    """
    if switch == "pri_asg":
        counter = (2 * NUM_REQUESTS) + (6 * NUM_NODES) + ((2 + NUM_PRIORITY_LEVELS) * (NUM_NODES ** 2))

    return counter


def parse_state(state, NUM_NODES, NUM_REQUESTS, env_obj, switch="none"):
    np.set_printoptions(suppress=True, linewidth=100)
    counter = 0

    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    """
    print("ACTIVE REQUESTS:")
    print(state[counter:NUM_REQUESTS].astype(int))
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST CAPACITY REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nPER NODE REQUEST BW REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nREQUEST DELAY REQUIREMENTS:")
    print(state[counter:counter + NUM_REQUESTS].astype(int))
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST BURST SIZES:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    if switch == "pri_asg":
        print("\nPER ASSIGNED NODE REQUEST BW REQUIREMENTS:")
        print(state[counter:counter + NUM_NODES].astype(int))
        counter += NUM_NODES

    print("\nDC CAPACITIES:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nDC COSTS:")
    print(state[counter:counter + NUM_NODES].astype(int))
    counter += NUM_NODES

    print("\nLINK BWS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].astype(int).reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK COSTS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].astype(int).reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK DELAYS MATRIX:")
    link_delays_matrix = state[counter:counter + env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)]. \
        reshape(env_obj.net_obj.NUM_PRIORITY_LEVELS, NUM_NODES, NUM_NODES)
    # since we removed null index 0, index 0 of link_delays_matrix is for priority 1 and so on.
    for n in range(0, env_obj.net_obj.NUM_PRIORITY_LEVELS):
        print(f"Priority: {n + 1}")
        print(link_delays_matrix[n])
    counter += env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)
    """

    print("ACTIVE REQUESTS:")
    print(state[counter:NUM_REQUESTS])
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST CAPACITY REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nPER NODE REQUEST BW REQUIREMENTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nREQUEST DELAY REQUIREMENTS:")
    print(state[counter:counter + NUM_REQUESTS])
    counter += NUM_REQUESTS

    print("\nPER NODE REQUEST BURST SIZES:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    if switch == "pri_asg":
        print("\nPER ASSIGNED NODE REQUEST BW REQUIREMENTS:")
        print(state[counter:counter + NUM_NODES])
        counter += NUM_NODES

    print("\nDC CAPACITIES:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nDC COSTS:")
    print(state[counter:counter + NUM_NODES])
    counter += NUM_NODES

    print("\nLINK BWS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK COSTS MATRIX:")
    print(state[counter:counter + NUM_NODES ** 2].reshape(NUM_NODES, NUM_NODES))
    counter += NUM_NODES ** 2

    print("\nLINK DELAYS MATRIX:")
    link_delays_matrix = state[counter:counter + env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)]. \
        reshape(env_obj.net_obj.NUM_PRIORITY_LEVELS, NUM_NODES, NUM_NODES)
    # since we removed null index 0, index 0 of link_delays_matrix is for priority 1 and so on.
    for n in range(0, env_obj.net_obj.NUM_PRIORITY_LEVELS):
        print(f"Priority: {n + 1}")
        print(link_delays_matrix[n])
    counter += env_obj.net_obj.NUM_PRIORITY_LEVELS * (NUM_NODES ** 2)

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")


def plot_learning_curve(x, y, epsilons, filename=""):
    fig = plt.figure()
    s_plt1 = fig.add_subplot(111, label="1")  # "234" means "2x3 grid, 4th subplot".
    s_plt2 = fig.add_subplot(111, label="2", frame_on=False)

    s_plt1.plot(x, epsilons, color="C0")
    s_plt1.set_xlabel("Game Number", color="C0")
    s_plt1.set_ylabel("Epsilon", color="C0")
    s_plt1.tick_params(axis="x", color="C0")
    s_plt1.tick_params(axis="y", color="C0")

    n = len(y)
    y_avg = np.empty(n)
    for i in range(n):
        y_avg[i] = np.mean(y[max(0, i - 100):(i + 1)])

    s_plt2.plot(x, y_avg, color="C1")
    s_plt2.axes.get_xaxis().set_visible(False)
    s_plt2.yaxis.tick_right()
    s_plt2.set_ylabel('Cost', color="C1")
    s_plt2.yaxis.set_label_position('right')
    s_plt2.tick_params(axis='y', colors="C1")

    plt.show()
    plt.savefig(filename)


def simple_plot(x, y, z, filename=""):
    fig = plt.figure()
    plt1 = fig.add_subplot(111, label="2")

    y_avg = np.empty(len(y))
    z_avg = np.empty(len(z))
    for i in range(len(y)):
        y_avg[i] = np.mean(y[max(0, i - 200):(i + 1)])
    for i in range(len(z)):
        z_avg[i] = np.mean(z[max(0, i - 200):(i + 1)])

    plt1.plot(x[100:], y_avg[100:], color="C1")
    plt1.plot(x[100:], z_avg[100:], color="C1")
    plt1.set_xlabel("Game Number", color="C1")
    plt1.set_ylabel("OF", color="C1")
    plt1.tick_params(axis="x", color="C1")
    plt1.tick_params(axis="y", color="C1")

    # plt.show()
    plt.savefig(filename)


def save_list_to_file(list, dir, file_name):
    full_name = dir + file_name + ".txt"
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    f = open(full_name, "w")
    for i in range(len(list)):
        if i < len(list) - 1:
            f.write(str(list[i]) + "\n")
        else:
            f.write(str(list[i]))
    f.close()


def read_list_from_file(dir, file_name, type, round_num=2):
    f = open(dir + file_name, "r")  # opens the file in read mode
    list = f.read().splitlines()  # puts the file into an array
    f.close()
    if type == "int":
        return [int(element) for element in list]
    if type == "float":
        return [round(float(element), round_num) for element in list]


def generate_seeds(num_seeds):
    seeds = []
    for i in range(num_seeds):
        seeds.append(np.random.randint(1, 100000))
    save_list_to_file(seeds, "inputs/", "SEEDS")
