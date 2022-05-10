from Environment import Environment
from Functions import parse_state, plot_learning_curve
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
best_score = -np.inf
load_checkpoint = False
EPSILON = 0 if load_checkpoint else 1
n_games = 1  # 1 for one sample
# NUM_ACTIONS = (NUM_NODES-len(env_obj.net_obj.get_first_tier_nodes()))
NUM_ACTIONS = NUM_NODES
file_name = str(NUM_NODES) + "_" + str(NUM_PRIORITY_LEVELS) + "_" + str(NUM_REQUESTS) + "_" + str(NUM_SERVICES) + "_" + str(n_games)
figure_file = file_name + '.png'
agent = Agent(EPSILON, NUM_ACTIONS, env_obj.get_state().size, file_name)

if load_checkpoint:
    agent.load_models()

n_steps = 0
scores, eps_history, steps, active_reqs, costs_sums = [], [], [], [], []

for i in range(n_games):
    SEED = 4  # np.random.randint(1, 1000)

    env_obj.reset(SEED)
    state = env_obj.get_state()

    score = 0
    n_active_reqs = 0
    costs_sum = 0
    for r in env_obj.req_obj.REQUESTS:
        ACTION_SEED = int(random.randrange(sys.maxsize)/(10 ** 15))
        # ACTION_SEED = 4
        a = {"req_id": r, "node_id": agent.choose_action(state, ACTION_SEED)}
        resulted_state, reward, done, info, OF = env_obj.step(a, "none")
        score += reward
        if not done:
            n_active_reqs += 1
        costs_sum += OF

        if not load_checkpoint:
            agent.store_transition(state, a["node_id"], reward, resulted_state, int(done))
            agent.learn()

        state = resulted_state
        n_steps += 1

    scores.append(score)
    steps.append(n_steps)
    active_reqs.append(n_active_reqs)
    costs_sums.append(costs_sum)
    avg_score = np.mean(scores[-100:])
    # print('episode:', i, 'score:', score, 'best_score: %.1f, eps: %.4f' % (best_score, agent.EPSILON), 'steps', n_steps)
    print('episode:', i, 'score:', score, 'active_reqs:', n_active_reqs, 'avg_costs_sum: %.1f, eps: %.4f' % (costs_sum/n_active_reqs, agent.EPSILON), 'steps', n_steps)

    if avg_score > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = avg_score

    eps_history.append(agent.EPSILON)

scores = np.array(scores)
x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps, scores/(10000*NUM_REQUESTS), eps_history, filename=figure_file)