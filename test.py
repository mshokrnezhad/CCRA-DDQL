"""
env_obj = Environment(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES)

best_score = -np.inf
load_checkpoint = False
EPSILON = 0 if load_checkpoint else 1
n_games = 500  # 1 for one sample
NUM_ACTIONS = (NUM_NODES-len(env_obj.net_obj.get_first_tier_nodes()))
file_name = str(NUM_NODES)+"_"+str(NUM_PRIORITY_LEVELS)+"_"+str(NUM_REQUESTS)+"_"+str(NUM_SERVICES)+"_"+str(n_games)
figure_file = file_name + '.png'

agent = Agent(GAMMA=0.99, EPSILON=EPSILON, LR=0.0001, NUM_ACTIONS=NUM_ACTIONS, INPUT_SHAPE=env_obj.get_state().size,
              MEMORY_SIZE=50000, BATCH_SIZE=32, EPSILON_MIN=0.1, EPSILON_DEC=1e-3, REPLACE_COUNTER=10000,
              NAME=file_name, CHECKPOINT_DIR='models/')

if load_checkpoint:
    agent.load_models()

n_steps = 0
scores, eps_history, steps_array = [], [], []

for i in range(n_games):
    done = False
    # SEED = np.random.randint(1, 1000)

    SEED = int(random.randrange(sys.maxsize)/(10 ** 15))
    env_obj.reset(SEED)
    state = env_obj.get_state()

    score = 0

    while not done:
        selected_request = env_obj.req_obj.REQUESTS.min()
        raw_action = agent.choose_action(state)
        # should be replaced by agent's action selection function
        # assigned_node = np.random.choice(np.setdiff1d(env_obj.net_obj.NODES, env_obj.net_obj.get_first_tier_nodes()))
        action = {"req_id": selected_request, "node_id": raw_action + len(env_obj.net_obj.get_first_tier_nodes())}

        resulted_state, reward, done, info = env_obj.step(action)
        # parse_state(resulted_state, NUM_NODES, NUM_REQUESTS, env_obj)
        # print(f"req: {action['req_id']}, assigned node: {action['node_id']}, reward: {reward}, status: {not done}")
        score += reward

        if not load_checkpoint:
            agent.store_transition(state, raw_action, reward, resulted_state, int(done))
            agent.learn()

        state = resulted_state
        n_steps += 1

    scores.append(score)
    steps_array.append(n_steps)

    avg_score = np.mean(scores[-100:])
    print('episode:', i, 'score:', score, 'best_score: %.1f, eps: %.4f'
          % (best_score, agent.EPSILON), 'steps', n_steps)
    # print('episode:', i, 'seed:', SEED, 'score:', score, 'best_score: %.1f, eps: %.4f'
    # % (best_score, agent.EPSILON), 'steps', n_steps)

    if avg_score > best_score:
        if not load_checkpoint:
            agent.save_models()
        best_score = avg_score

    eps_history.append(agent.EPSILON)

scores = np.array(scores)
x = [i+1 for i in range(len(scores))]
plot_learning_curve(steps_array, scores/(10000*NUM_REQUESTS), eps_history, filename=figure_file)
"""

"""
from Environment import Environment
from Functions import parse_state, plot_learning_curve
from Agent import Agent
import numpy as np
import sys
import random

NUM_NODES = 12
NUM_PRIORITY_LEVELS = 1
NUM_REQUESTS = 10
NUM_SERVICES = 1
env_obj = Environment(NUM_NODES, NUM_PRIORITY_LEVELS, NUM_REQUESTS, NUM_SERVICES)

SEED = np.random.randint(1, 1000)
env_obj.reset(SEED)
opt_results = env_obj.heu_obj.solve()
print(opt_results)
"""

import numpy as np; np.random.seed(43)
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset

#  random  walk
random_walk = np.cumsum(np.random.randn(5000))/2-50

fig = plt.figure(figsize = (13,8))
ax = plt.subplot(111) #whole path
ax.plot(random_walk)
ax.set_xlim(0,5000)
ax.set_ylim(-130,55)

axins = zoomed_inset_axes(ax,2,loc='lower right',
                          axes_kwargs={"facecolor" : "lightgray"})

axins.plot(random_walk)

x1,x2,y1,y2 = 1000,2000, 1600,1700
axins.set_xlim(x1,x2)
axins.set_ylim(y1,y2)

pp,p1,p2 = mark_inset(ax,axins,loc1=1,loc2=3)
pp.set_fill(True)
pp.set_facecolor("lightgray")
pp.set_edgecolor("k")
plt.show()
