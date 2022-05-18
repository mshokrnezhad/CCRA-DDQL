AGENT_EPSILON_MIN=0.01

dc_cost_unit = np.array([(self.DC_COST_RATIO ** (self.NUM_TIERS - self.get_tier_num(i) - 1)) * self.DC_COST_BASE for i in self.NODES]) # DC_COST_RATIO=10, DC_COST_BASE=1000
dc_costs = np.array([rnd.randint(dc_cost_unit[i] - 5, dc_cost_unit[i] + 5) for i in self.NODES])

seeds.append(np.random.randint(1, 100000))