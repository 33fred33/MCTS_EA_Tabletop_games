import random as rd
from Agents.base_agents import BaseAgent
import pandas as pd
import Utilities.logs_management as lm

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
    
    def dump_my_logs(self, file_path, file_name):
        for logs in [self.agent_data(), self.choose_action_logs]:
            lm.dump_data(logs, file_path, file_name)

    def __str__(self):
        return self.name