from typing import Protocol
from enum import Enum, auto
from typing import List

class BaseGameState(Protocol):

    def make_action(self, action) -> None: raise NotImplementedError

    def is_terminal(self) -> bool: raise NotImplementedError

    def duplicate(self): raise NotImplementedError

    def set_initial_state(self) -> None: raise NotImplementedError

    def winner(self) -> int: raise NotImplementedError

    def feature_vector(self) -> List: raise NotImplementedError

    available_actions: List #List[Action]
    player_turn:int #begins in 0, ends in n_players-1
    score:List[float] #Default:None. Updated to reflect the final score
    turn:int #begins in 1
