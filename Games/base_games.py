from typing import Protocol
from enum import Enum, auto
from typing import List
import pandas as pd

class BaseGameState(Protocol):

    def make_action(self, action) -> None: raise NotImplementedError

    def duplicate(self): raise NotImplementedError

    def set_initial_state(self) -> None: raise NotImplementedError

    def winner(self) -> int: raise NotImplementedError

    def feature_vector(self) -> List: raise NotImplementedError

    def logs_data(self) -> pd.DataFrame: raise NotImplementedError

    def game_definition_data(self) -> pd.DataFrame: raise NotImplementedError

    available_actions: List #List[Action]
    player_turn:int #begins in 0, ends in n_players-1
    reward:List[float] #Reward given to each player at any time-step. Default:None. Updated to reflect the reward of the last action
    turn:int #begins in 1
    is_terminal:bool #Default:False. Updated to True when the game is over
    name:str #Name of the game
