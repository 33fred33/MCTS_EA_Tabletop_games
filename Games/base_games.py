from typing import Protocol
from enum import Enum, auto

class BaseGameState(Protocol):

    def make_action(self, action) -> None: raise NotImplementedError

    def is_terminal(self) -> bool: raise NotImplementedError

    def duplicate(self): raise NotImplementedError

    def set_initial_state(self) -> None: raise NotImplementedError

    def winner(self) -> int: raise NotImplementedError

    available_actions:dict
    player_turn:int

class BaseAction(Protocol):
    
    index: int