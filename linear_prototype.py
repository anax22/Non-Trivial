# this is the code for a linear prototype for Hierarchical Value Alignment in MAS
# it uses a static environment, an adaptive environment will be implemented in later prototypes
# as well as reinforcemnt learning for adapting values
# now thye just use formulas described in the paper
from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
import random

class CarAgent(Agent):
    def __init__(self, unique_id, model): #initializes the car agent
        super().__init__(unique_id, model)
        # uses random values for the level of alignment of each goal, in reality these wouldn't be random
        # they would be predetermined and different for each system/model 
        self.high_level_goals = {"Safety": random.uniform(0, 1), "Efficiency": random.uniform(0, 1)}
        self.mid_level_goals = {"TrafficManagement": random.uniform(0, 1), "RouteOptimization": random.uniform(0, 1)}
        self.low_level_goals = {"Acceleration": random.uniform(0, 1), "Braking": random.uniform(0, 1)}
        # set constants
        self.alpha = 0.1
        self.beta = 0.05

    # this function ensures communication between agents at all levels
    def communicate(self, other_agent):
        self.align_high_level_goals(other_agent.high_level_goals)
        self.align_mid_level_goals(other_agent.mid_level_goals)
        self.align_low_level_goals(other_agent.low_level_goals)

    def align_high_level_goals(self, other_high_level_goals):
        for goal in self.high_level_goals: # align with goals at the same level
            self.high_level_goals[goal] += (other_high_level_goals[goal] - self.high_level_goals[goal]) * self.alpha
        # these are the highest level goals, so there isn't a higher level they can influence

    def align_mid_level_goals(self, other_mid_level_goals):
        for goal in self.mid_level_goals:
            self.mid_level_goals[goal] += (other_mid_level_goals[goal] - self.mid_level_goals[goal]) * self.alpha
             # updates high level goals based on mid-level goals as well, since the whole system needs to be aligned
            for goal2 in self.high_level_goals:
                self.high_level_goals[goal2] += (self.mid_level_goals[goal] - self.high_level_goals[goal2]) * self.beta

    def align_low_level_goals(self, other_low_level_goals):
        for goal in self.low_level_goals:
            self.low_level_goals[goal] += (other_low_level_goals[goal] - self.low_level_goals[goal]) * self.alpha
            # updates mid level goals based on low-level values
            for goal2 in self.mid_level_goals:
                self.mid_level_goals[goal2] += (self.low_level_goals[goal] - self.mid_level_goals[goal2]) * self.beta

    def step(self):
        neighbors = self.model.schedule.agents # gathers all agents in neighbors
        for neighbor in neighbors:
            if neighbor != self and isinstance(neighbor, CarAgent): # for each one that isn't itself
                self.communicate(neighbor) # align goals based on the other
        
        # print progress made
        print(f"\nData after step {self.model.schedule.steps} for Agent {self.unique_id}:\n")
        print("High Level goals:", self.high_level_goals)
        print("Mid Level goals:", self.mid_level_goals)
        print("Low Level goals:", self.low_level_goals)

class CarModel(Model): # initialize car model with the set number of agents
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.schedule = BaseScheduler(self)

        for i in range(self.num_agents):
            agent = CarAgent(i, self)
            self.schedule.add(agent)

    def step(self):
        self.schedule.step()

# run simulation
model = CarModel(num_agents = 3)
for i in range(5):
    model.step()
