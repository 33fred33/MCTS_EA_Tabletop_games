import random as rd
from Agents.base_agents import BaseAgent
import pandas as pd
import Utilities.logs_management as lm
import numpy as np

class OSLA_Wins(BaseAgent):
    
    player : int = 0

    def __init__(self, name="OSLA_Wins", logs = False, max_probing_size = np.inf):
        self.name = name
        self.logs = logs
        self.choose_action_logs = pd.DataFrame({"Player":self.name, "player_name":self.name}, index=[0])
        self.isAIPlayer = True
        self.current_fm = 0
        self.max_probing_size = max_probing_size
    
    def choose_action(self, state):
        #Finds immediate wins, and returns that action. Random otherwise.
        self.current_fm = 0
        self.player = state.player_turn
        probing_size = self.max_probing_size + 1
        
        if probing_size < len(state.available_actions):
            probing_limited = True
            actions_to_test = rd.sample(state.available_actions, probing_size)
        else: 
            probing_limited = False
            actions_to_test = state.available_actions

        for action in actions_to_test:
            duplicate_state = state.duplicate()
            duplicate_state.make_action(action)
            self.current_fm += 1
            if duplicate_state.is_terminal:
                if duplicate_state.winner == self.player:
                    return action
        if probing_limited:
            return actions_to_test[-1]
        return rd.choice(state.available_actions)

    def agent_data(self):
        return pd.DataFrame({"Player":str(self), "player_name":self.name, "max_probing_size":self.max_probing_size}, index=[0])
    
    def dump_my_logs(self, file_path, file_name):
        for logs in [self.agent_data(), self.choose_action_logs]:
            lm.dump_data(logs, file_path, file_name)

    def __str__(self):
        return self.name