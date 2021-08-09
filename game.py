from abc import ABC, abstractmethod
from enum import Enum, auto

class Players(Enum):

    PLAYER1 = auto()
    PLAYER2 = auto()
    
    def succ(self):

        if self == Players.PLAYER1:
            return Players.PLAYER2
        else:
            return Players.PLAYER1

class BaseGameState(ABC):

    @abstractmethod 
    def make_action(self, action): pass

    @abstractmethod 
    def is_terminal(self): pass

    @abstractmethod
    def duplicate(self): pass

    @abstractmethod 
    def set_initial_state(self): pass

    @abstractmethod 
    def winner(self): pass

    @property
    @abstractmethod 
    def available_actions(self):
        raise NotImplementedError

    @property
    @abstractmethod 
    def make_action_calls(self):
        raise NotImplementedError

    @property
    @abstractmethod 
    def player_turn(self):
        raise NotImplementedError

        

class BaseAction(ABC):

    @abstractmethod 
    def duplicate(self): pass

class Agent(ABC):

    @abstractmethod 
    def choose_action(self): pass

    @property
    @abstractmethod 
    def player(self):
        raise NotImplementedError