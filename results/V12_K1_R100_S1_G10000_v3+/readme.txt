dc_cost_unit = np.array([(self.DC_COST_RATIO ** (self.NUM_TIERS - self.get_tier_num(i) - 1)) * self.DC_COST_BASE for i in self.NODES]) # DC_COST_RATIO=10, DC_COST_BASE=1000
dc_costs = np.array([rnd.randint(dc_cost_unit[i] - 5, dc_cost_unit[i] + 5) for i in self.NODES])

norm = 1 # no normalization

BURST_SIZE_LB=1
BURST_SIZE_UB=2

BURST_SIZE_LIMIT=200,
PACKET_SIZE=1

seeds.append(np.random.randint(1, 100))

DC_COST_BASE=1
DC_COST_RATIO=50
dc_cost_unit = np.array([(self.DC_COST_RATIO ** (self.NUM_TIERS - self.get_tier_num(i))) * self.DC_COST_BASE for i in self.NODES])
dc_costs = np.array([rnd.randint(dc_cost_unit[i] - 5, dc_cost_unit[i] + 5) for i in self.NODES])

DC_CAPACITY_UNIT=250
dc_capacities = np.array([rnd.randint(self.DC_CAPACITY_UNIT, self.DC_CAPACITY_UNIT + 1) for i in self.NODES])

max_cost = self.net_obj.MAX_COST_PER_TIER[result["pair"][0]][self.net_obj.get_tier_num(result["pair"][1])]
min_cost = self.net_obj.MIN_COST_PER_TIER[result["pair"][0]][self.net_obj.get_tier_num(result["pair"][1])]
tier_cost_range = max_cost - min_cost
action_efficiency_range = 100
action_efficiency = action_efficiency_range - (action_efficiency_range * (result["OF"] - min_cost) / tier_cost_range)
reward_base = 100
reward = reward_base ** (self.net_obj.get_tier_num(result["pair"][1]) + 1) + action_efficiency
self.update_state(action, result)

EPSILON_MIN=0,
EPSILON_DEC=5e-6,

1: Delay Requirement: 0-1, Capacity Requirement: 4-8
2: Delay Requirement: 2-3, Capacity Requirement: 4-8
3: Delay Requirement: 10-11, Capacity Requirement: 4-8