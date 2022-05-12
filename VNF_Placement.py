from Environment import Environment
from Agent import Agent
import numpy as np
import random
import sys
from Functions import parse_state, plot_learning_curve, calculate_input_shape


class VNF_Placement(object):
    def __init__(self, NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS, NUM_GAMES):
        self.SWITCH = "vnf_plc"
        self.NUM_NODES = NUM_NODES
        self.NUM_REQUESTS = NUM_REQUESTS
        self.NUM_SERVICES = NUM_SERVICES
        self.NUM_PRIORITY_LEVELS = NUM_PRIORITY_LEVELS
        self.NUM_GAMES = NUM_GAMES
        self.NUM_ACTIONS = NUM_NODES
        self.FILE_NAME = "V" + str(NUM_NODES) + "_K" + str(NUM_PRIORITY_LEVELS) + "_R" + str(NUM_REQUESTS) + "_S" + str(NUM_SERVICES) + "_G" + str(NUM_GAMES)
        self.env_obj = Environment(NUM_NODES, NUM_REQUESTS, NUM_SERVICES, NUM_PRIORITY_LEVELS)
        self.agent = Agent(self.NUM_ACTIONS, self.env_obj.get_state().size, self.FILE_NAME)

    def DDQL(self):
        print("Aaaaaaa")
        """
        best_reward = -np.inf
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

    def test(self, TEST_SIZE=100):
        try:
            self.agent.load_models()

            best_score = -np.inf
            n_steps = 0
            scores, eps_history, steps_array = [], [], []

            for i in range(TEST_SIZE):
                done = False
                score = 0
                SEED = int(random.randrange(sys.maxsize) / (10 ** 15))

                self.env_obj.reset(SEED)
                state = self.env_obj.get_state(self.SWITCH)

                while not done:
                    selected_request = self.env_obj.req_obj.REQUESTS.min()
                    raw_action = self.agent.choose_action(state, False)
                    action = {"req_id": selected_request,
                              "node_id": raw_action + len(self.env_obj.net_obj.get_first_tier_nodes())}

                    resulted_state, reward, done, info = self.env_obj.step(action, "srv_plc")

                    score += reward

                    state = resulted_state
                    n_steps += 1

                scores.append(score)
                steps_array.append(n_steps)

                avg_score = np.mean(scores[-100:])
                print('episode:', i, 'score:', score, 'avg_score: %.1f' % (avg_score), 'steps', n_steps)

                if avg_score > best_score:
                    best_score = avg_score

                eps_history.append(self.agent.EPSILON)

            scores = np.array(scores)
            x = [i + 1 for i in range(len(scores))]
            plot_learning_curve(steps_array, scores / (10000 * self.NUM_REQUESTS), eps_history,
                                filename=self.FILE_NAME + '_test.png')

        except:
            print("there is no trained model for the input scenario!")

    def get_allocations(self, env_obj):
        try:
            assigned_nodes = []
            self.agent.load_models()

            # env_obj.reset(SEED)
            state = self.env_obj.get_state(self.SWITCH)

            for i in range(self.NUM_REQUESTS):
                raw_action = self.agent.choose_action(state, False)
                action = {"req_id": i, "node_id": raw_action + len(self.env_obj.net_obj.get_first_tier_nodes())}
                assigned_nodes.append(raw_action + len(self.env_obj.net_obj.get_first_tier_nodes()))
                # print(action)

                result = self.env_obj.model_obj.solve(action)

                if result["done"]:
                    print("there is no feasible solution for the input scenario!")
                    break
                else:
                    self.env_obj.update_state(action, result)
                    state = self.env_obj.get_state(self.SWITCH)

            # print(f'\nAssigned Nodes: {assigned_nodes}')
            return assigned_nodes

        except:
            print("there is no trained model for the input scenario!")
            return -1