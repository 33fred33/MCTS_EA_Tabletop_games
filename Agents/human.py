import random as rd
from Agents.base_agents import BaseAgent
import pandas as pd
import Utilities.logs_management as lm

class DefPlayer:
    
    def __init__(self, player_index=0):
        self.isFirstPlayer = player_index
        self.name = "No Name"
        self.logfile = None
        self.fullName = "Definitely has no name"
        self.isAIPlayer = True
        self.family = None
        self.opponent = None   
    
    def choose_action(self,state):
        """
        Move choice based on type of player
        """
        pass
    
    def __repr__(self):
        return self.name

class HumanPlayer(DefPlayer):
    
    def __init__(self, name = 'Human'):
        super().__init__()
        self.name = name
        self.fullName = "Human Player"
        self.isAIPlayer = False
        self.family = "Human"
    
    def choose_action(self, state):
        """
        state - The current state of the game board
        """
        positions = state.available_actions()
        
        # user input
        while True:
            print(f'Available moves: \n {positions} \n')
            choice = int(input("Input your choice:"))
            if choice in positions:
                return choice