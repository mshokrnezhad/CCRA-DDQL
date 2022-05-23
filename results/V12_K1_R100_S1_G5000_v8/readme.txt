dc_cost_unit = np.array([(self.DC_COST_RATIO ** (self.NUM_TIERS - self.get_tier_num(i) - 1)) * self.DC_COST_BASE for i in self.NODES]) # DC_COST_RATIO=10, DC_COST_BASE=1000
dc_costs = np.array([rnd.randint(dc_cost_unit[i] - 5, dc_cost_unit[i] + 5) for i in self.NODES])

norm = 1 # no normalization

CAPACITY_REQUIREMENT_LB=1
CAPACITY_REQUIREMENT_UB=4
BW_REQUIREMENT_LB=2
BW_REQUIREMENT_UB=3
BURST_SIZE_LB=1
BURST_SIZE_UB=2

EPSILON_MIN=0,
EPSILON_DEC=5e-6

BURST_SIZE_LIMIT=200,
PACKET_SIZE=1

seeds.append(np.random.randint(1, 100))