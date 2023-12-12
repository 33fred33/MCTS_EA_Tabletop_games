from typing_extensions import Protocol
from enum import Enum, auto
from typing import List
import pandas as pd

class RandomEvent():
    def __init__(self, id, probability=None, event_type = None):
        self.probability = probability
        self.id = id
        self.event_type = event_type

    def duplicate(self):
        return RandomEvent(self.id, self.probability, self.event_type)
            
    def __repr__(self):
        String = "RandomEvent id:"+str(self.id)+", prob:"+"{0:.3g}".format(self.probability) +" type:"+str(self.event_type)
        return String
    
    def __eq__(self, other: object) -> bool:
        if self.id == other.id and self.event_type == other.event_type:
            return True
        else:
            return False
    
    def __hash__(self) -> int:
        return hash((self.id, self.event_type))

class BaseGameState(Protocol):

    def make_action(self, action) -> None: raise NotImplementedError

    def duplicate(self): raise NotImplementedError

    def set_initial_state(self) -> None: raise NotImplementedError

    def winner(self) -> int: raise NotImplementedError

    def feature_vector(self) -> dict: raise NotImplementedError

    def logs_data(self) -> pd.DataFrame: raise NotImplementedError

    def game_definition_data(self) -> pd.DataFrame: raise NotImplementedError

    available_actions: List #List[Action]
    player_turn:int #begins in 0, ends in n_players-1
    players:int #Number of players
    reward:List[float] #Reward given to each player at any time-step. Default:None. Updated to reflect the reward of the last action
    turn:int #begins in 1
    is_terminal:bool #Default:False. Updated to True when the game is over
    name:str #Name of the game
    losing_reward:float #Default:0. Reward given to a player when he loses
    draw_reward:float #Default:0. Reward given to a player when the game ends in a draw
    winning_reward:float #Default:1. Reward given to a player when he wins

    #For random games:
    random_events:List #List[RandomEvent] -> listed in case they were sequencial
    def sample_random_event(self, event_type) -> RandomEvent: raise NotImplementedError #Samples a random event again, updating the game state and available_actions to accommodate it accordingly 
