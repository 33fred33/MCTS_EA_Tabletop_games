import random as rd
from Agents.base_agents import BaseAgent

class RandomPlayer(BaseAgent):
    
    player : int = 0

    def __init__(self, name="Random"):
        self.name = name
    
    def choose_action(self, state): #game interface dependencies
        return rd.choice(state.available_actions)
    
    def __str__(self):
        return self.name