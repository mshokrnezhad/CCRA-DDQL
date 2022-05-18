from Environment import Environment
from Agent import Agent
import numpy as np
import random
import sys
from Functions import parse_state, plot_learning_curve, calculate_input_shape, save_list_to_file, simple_plot
import numpy as np
import matplotlib.pyplot as plt
rnd = np.random


class VNF_Placement(object):
    def __init__(self, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS, NUM_GAMES, SEEDS):
        self.SWITCH = "vnf_plc"
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.NUM_GAMES = NUM_GAMES
        self.NUM_ACTIONS = NUM_NODES
        self.SEEDS = SEEDS
        self.FILE_NAME = "V" + str(NUM_NODES) + "_K" + str(NUM_PRIORITY_LEVELS) + "_R" + str(NUM_REQUESTS) + "_S" + str(NUM_SERVICES) + "_G" + str(NUM_GAMES)
        self.env_obj = Environment(NUM_NODES=NUM_NODES, NUM_REQUESTS=NUM_REQUESTS, NUM_SERVICES=NUM_SERVICES, NUM_PRIORITY_LEVELS=NUM_PRIORITY_LEVELS)
        self.agent = Agent(NUM_ACTIONS=self.NUM_ACTIONS, INPUT_SHAPE=self.env_obj.get_state().size, NAME=self.FILE_NAME)

    def ddql_alloc_train(self):
        best_reward = -np.inf
        num_steps = 0
        rewards, epsilons, steps, ml_nums_act_reqs, ml_avg_ofs = [], [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            state = self.env_obj.get_state()
            game_reward = 0
            ml_game_num_act_reqs = 0
            ml_game_of = 0
            for r in self.env_obj.req_obj.REQUESTS:
                ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))  # 4
                a = {"req_id": r, "node_id": self.agent.choose_action(state, ACTION_SEED)}
                resulted_state, req_reward, done, info, req_of = self.env_obj.step(a, "none")
                game_reward += req_reward
                if not done:
                    ml_game_num_act_reqs += 1
                ml_game_of += req_of
                self.agent.store_transition(state, a["node_id"], req_reward, resulted_state, int(done))
                self.agent.learn()
                state = resulted_state
                num_steps += 1
                # print(a["node_id"], req_reward)
            rewards.append(game_reward)
            steps.append(num_steps)
            ml_nums_act_reqs.append(ml_game_num_act_reqs)
            ml_avg_game_of = 0 if ml_game_num_act_reqs == 0 else ml_game_of / ml_game_num_act_reqs
            ml_avg_ofs.append(ml_avg_game_of)
            avg_reward = np.mean(rewards[-100:])
            epsilons.append(self.agent.EPSILON)
            if avg_reward > best_reward:
                self.agent.save_models()
                best_reward = avg_reward
            print('episode:', i, 'cost: %.3f, num_act_reqs: %.0f, reward: %.0f, eps: %.4f' % (ml_avg_game_of, ml_game_num_act_reqs, game_reward, self.agent.EPSILON), 'steps:', num_steps)

        save_list_to_file(ml_nums_act_reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_nums_act_reqs")
        save_list_to_file(ml_avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_avg_ofs")
        save_list_to_file(rewards, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_rewards")
        save_list_to_file(epsilons, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_epsilons")

        # simple_plot(range(self.NUM_GAMES), ml_avg_ofs, filename="results/" + self.FILE_NAME + "/" + self.FILE_NAME + "_ml_avg_ofs" + '.png')

    def ddql_alloc_eval(self):
        self.agent.load_models()
        self.agent.EPSILON = 0

        best_reward = -np.inf
        num_steps = 0
        rewards, epsilons, steps, ml_nums_act_reqs, ml_avg_ofs = [], [], [], [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            state = self.env_obj.get_state()
            game_reward = 0
            ml_game_num_act_reqs = 0
            ml_game_of = 0
            for r in self.env_obj.req_obj.REQUESTS:
                ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))  # 4
                a = {"req_id": r, "node_id": self.agent.choose_action(state, ACTION_SEED)}
                resulted_state, req_reward, done, info, req_of = self.env_obj.step(a, "none")
                game_reward += req_reward
                if not done:
                    ml_game_num_act_reqs += 1
                ml_game_of += req_of
                state = resulted_state
                num_steps += 1
                # print(a["node_id"], req_reward)
            rewards.append(game_reward)
            steps.append(num_steps)
            ml_nums_act_reqs.append(ml_game_num_act_reqs)
            ml_avg_game_of = 0 if ml_game_num_act_reqs == 0 else ml_game_of / ml_game_num_act_reqs
            ml_avg_ofs.append(ml_avg_game_of)
            avg_reward = np.mean(rewards[-100:])
            epsilons.append(self.agent.EPSILON)
            if avg_reward > best_reward:
                best_reward = avg_reward
            print('episode:', i, 'cost: %.3f, num_act_reqs: %.0f, eps: %.4f' % (ml_avg_game_of, ml_game_num_act_reqs, self.agent.EPSILON), 'steps:', num_steps)

        save_list_to_file(ml_nums_act_reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_nums_act_reqs_eval")
        save_list_to_file(ml_avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_ml_avg_ofs_eval")
        save_list_to_file(rewards, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_rewards_eval")

        # simple_plot(range(self.NUM_GAMES), ml_avg_ofs, filename="results/" + self.FILE_NAME + "/" + self.FILE_NAME + "_ml_avg_ofs" + '.png')

    def wf_alloc(self):
        opt_nums_act_reqs, opt_avg_ofs = [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            opt_results = self.env_obj.heu_obj.solve()
            # print(opt_results["pairs"])
            opt_game_num_act_reqs = opt_results["num_act_reqs"]
            opt_nums_act_reqs.append(opt_game_num_act_reqs)
            opt_avg_game_of = opt_results["avg_of"]
            opt_avg_ofs.append(opt_avg_game_of)
            print('episode:', i, 'cost: %.0f' % opt_avg_game_of)

        save_list_to_file(opt_nums_act_reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_opt_nums_act_reqs")
        save_list_to_file(opt_avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_opt_avg_ofs")

        # simple_plot(range(self.NUM_GAMES), opt_avg_ofs, filename="results/" + self.FILE_NAME + "/" + self.FILE_NAME + "_opt_avg_ofs" + '.png')

    def rnd_alloc(self):
        rnd_nums_act_reqs, rnd_avg_ofs = [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            rnd_game_num_act_reqs = 0
            rnd_game_of = 0
            for r in self.env_obj.req_obj.REQUESTS:
                ACTION_SEED = int(random.randrange(sys.maxsize) / (10 ** 15))  # 4
                rnd.seed(ACTION_SEED)
                a = {"req_id": r, "node_id": rnd.choice(self.env_obj.net_obj.NODES)}
                resulted_state, req_reward, done, info, req_of = self.env_obj.step(a, "none")
                if not done:
                    rnd_game_num_act_reqs += 1
                rnd_game_of += req_of
            rnd_nums_act_reqs.append(rnd_game_num_act_reqs)
            rnd_avg_game_of = 0 if rnd_game_num_act_reqs == 0 else rnd_game_of / rnd_game_num_act_reqs
            rnd_avg_ofs.append(rnd_avg_game_of)
            print('episode:', i, 'cost: %.0f, num_act_reqs: %.0f' % (rnd_avg_game_of, rnd_game_num_act_reqs))

        save_list_to_file(rnd_nums_act_reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_rnd_nums_act_reqs")
        save_list_to_file(rnd_avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_rnd_avg_ofs")

    """
    def random_alloc(self):  # everything is random
        random_nums_act_reqs, random_avg_ofs = [], []

        for i in range(self.NUM_GAMES):
            SEED = self.SEEDS[i]
            self.env_obj.reset(SEED)
            
            random_results = self.env_obj.random_obj.solve()
            # print(opt_results["pairs"])
            random_game_num_act_reqs = random_results["num_act_reqs"]
            random_nums_act_reqs.append(random_game_num_act_reqs)
            random_avg_game_of = random_results["avg_of"]
            random_avg_ofs.append(random_avg_game_of)
            print('episode:', i, 'cost: %.0f, num_act_reqs: %.0f' %(random_avg_game_of, random_game_num_act_reqs))

        save_list_to_file(random_nums_act_reqs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_random_nums_act_reqs")
        save_list_to_file(random_avg_ofs, "results/" + self.FILE_NAME + "/", self.FILE_NAME + "_random_avg_ofs")

        # simple_plot(range(self.NUM_GAMES), opt_avg_ofs, filename="results/" + self.FILE_NAME + "/" + self.FILE_NAME + "_opt_avg_ofs" + '.png')
    """