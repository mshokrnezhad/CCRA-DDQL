dc_capacities = np.array([rnd.randint(self.get_tier_num(i) * self.DC_CAPACITY_UNIT, (self.get_tier_num(i) + 1) * self.DC_CAPACITY_UNIT) for i in self.NODES]) # self.DC_CAPACITY_UNIT = 50

seeds.append(np.random.randint(1, 100000))