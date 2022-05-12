from Environment import Environment
from Functions import parse_state, plot_learning_curve, save_list_to_file
from Agent import Agent
import numpy as np
import sys
import random

# from Service_Placement import Service_Placement
# from Priority_Assignment import Priority_Assignment

NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 10
NUM_SERVICES = 1
env_obj = Environment(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES)
best_reward = -np.inf
load_checkpoint = False
EPSILON = 0 if load_checkpoint else 1
num_games = 10
# NUM_ACTIONS = (NUM_NODES-len(env_obj.net_obj.get_first_tier_nodes()))
NUM_ACTIONS = NUM_NODES
file_name = "V" + str(NUM_NODES) + "_K" + str(NUM_PRIORITY_LEVELS) + "_R" + str(NUM_REQUESTS) + "_S" + str(NUM_SERVICES) + "_G" + str(num_games)
figure_file = file_name + '.png'
agent = Agent(EPSILON, NUM_ACTIONS, env_obj.get_state().size, file_name)

if load_checkpoint:
    agent.load_models()

num_steps = 0
rewards, epsilons, steps, ml_nums_act_reqs, ml_avg_ofs, opt_nums_act_reqs, opt_avg_ofs, accuracies = [], [], [], [], [], [], [], []

for i in range(num_games):
    SEED = np.random.randint(1, 1000)  # 4

    env_obj.reset(SEED)
    state = env_obj.get_state()
    game_reward = 0
    ml_game_num_act_reqs = 0
    ml_game_of = 0
    for r in env_obj.req_obj.REQUESTS:
        ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))  # 4
        a = {"req_id": r, "node_id": agent.choose_action(state, ACTION_SEED)}
        resulted_state, req_reward, done, info, req_of = env_obj.step(a, "none")
        game_reward += req_reward
        if not done:
            ml_game_num_act_reqs += 1
        ml_game_of += req_of
        if not load_checkpoint:
            agent.store_transition(state, a["node_id"], req_reward, resulted_state, int(done))
            agent.learn()
        state = resulted_state
        num_steps += 1
    rewards.append(game_reward)
    steps.append(num_steps)
    ml_nums_act_reqs.append(ml_game_num_act_reqs)
    ml_avg_game_of = 0 if ml_game_num_act_reqs == 0 else ml_game_of / ml_game_num_act_reqs
    ml_avg_ofs.append(ml_avg_game_of)
    avg_reward = np.mean(rewards[-100:])
    epsilons.append(agent.EPSILON)
    if avg_reward > best_reward:
        if not load_checkpoint:
            agent.save_models()
        best_reward = avg_reward

    env_obj.reset(SEED)
    opt_results = env_obj.heu_obj.solve()
    opt_game_num_act_reqs = opt_results["num_act_reqs"]
    opt_nums_act_reqs.append(opt_game_num_act_reqs)
    opt_avg_game_of = opt_results["avg_of"]
    opt_avg_ofs.append(opt_avg_game_of)

    accuracy = 1 - (abs(ml_avg_game_of - opt_avg_game_of) / opt_avg_game_of)
    accuracies.append(accuracy)

    print('episode:', i, 'acc: %.3f, best_reward: %.0f, eps: %.4f' % (accuracy, best_reward, agent.EPSILON), 'steps:', num_steps)

save_list_to_file(opt_nums_act_reqs, "results/" + file_name + "/", file_name, "opt_nums_act_reqs")
save_list_to_file(opt_avg_ofs, "results/" + file_name + "/", file_name, "opt_avg_ofs")
save_list_to_file(ml_nums_act_reqs, "results/" + file_name + "/", file_name, "ml_nums_act_reqs")
save_list_to_file(ml_avg_ofs, "results/" + file_name + "/", file_name, "ml_avg_ofs")
save_list_to_file(rewards, "results/" + file_name + "/", file_name, "rewards")
save_list_to_file(epsilons, "results/" + file_name + "/", file_name, "epsilons")
save_list_to_file(accuracies, "results/" + file_name + "/", file_name, "opt_vs_ml")

"""
rewards = np.array(rewards)
x = [i + 1 for i in range(len(rewards))]
plot_learning_curve(range(num_games), ml_avg_ofs, epsilons, filename=figure_file)
"""