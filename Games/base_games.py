from typing import Protocol
from enum import Enum, auto
from typing import List

class BaseGameState(Protocol):

    def make_action(self, action) -> None: raise NotImplementedError

    def is_terminal(self) -> bool: raise NotImplementedError

    def duplicate(self): raise NotImplementedError

    def set_initial_state(self) -> None: raise NotImplementedError

    def winner(self) -> int: raise NotImplementedError

    available_actions: List #List[Action]
    player_turn:int 
