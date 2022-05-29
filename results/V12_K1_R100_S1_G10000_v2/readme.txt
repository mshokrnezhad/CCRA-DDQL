dc_cost_unit = np.array([(self.DC_COST_RATIO ** (self.NUM_TIERS - self.get_tier_num(i) - 1)) * self.DC_COST_BASE for i in self.NODES]) # DC_COST_RATIO=10, DC_COST_BASE=1000
dc_costs = np.array([rnd.randint(dc_cost_unit[i] - 5, dc_cost_unit[i] + 5) for i in self.NODES])

norm = 1 # no normalization

CAPACITY_REQUIREMENT_LB=1
CAPACITY_REQUIREMENT_UB=4
BW_REQUIREMENT_LB=2
BW_REQUIREMENT_UB=3
BURST_SIZE_LB=1
BURST_SIZE_UB=2

BURST_SIZE_LIMIT=200,
PACKET_SIZE=1

seeds.append(np.random.randint(1, 100))

DC_COST_BASE=1
DC_COST_RATIO=10
dc_cost_unit = np.array([(self.DC_COST_RATIO ** (self.NUM_TIERS - self.get_tier_num(i))) * self.DC_COST_BASE for i in self.NODES])
dc_costs = np.array([rnd.randint(dc_cost_unit[i] - 5, dc_cost_unit[i] + 5) for i in self.NODES])

max_cost = self.net_obj.MAX_COST_PER_TIER[result["pair"][0]][self.net_obj.get_tier_num(result["pair"][1])]
min_cost = self.net_obj.MIN_COST_PER_TIER[result["pair"][0]][self.net_obj.get_tier_num(result["pair"][1])]
tier_cost_range = max_cost - min_cost
action_efficiency_range = 100
action_efficiency = action_efficiency_range - (action_efficiency_range * (result["OF"] - min_cost) / tier_cost_range)
reward_base = 1000
reward = reward_base * (self.net_obj.get_tier_num(result["pair"][1]) + 1) + action_efficiency
self.update_state(action, result)