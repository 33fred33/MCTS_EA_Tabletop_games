import random as rd
from Agents.base_agents import BaseAgent
import pandas as pd

class RandomPlayer(BaseAgent):
    
    player : int = 0

    def __init__(self, name="Random", logs = False):
        self.name = name
        self.logs = logs
        self.choose_action_logs = pd.DataFrame({"Player":self.name, "player_name":self.name}, index=[0])
    
    def choose_action(self, state): #game interface dependencies
        return rd.choice(state.available_actions)

    def agent_data(self):
        return pd.DataFrame({"Player":str(self), "player_name":self.name}, index=[0])
    
    def __str__(self):
        return self.name