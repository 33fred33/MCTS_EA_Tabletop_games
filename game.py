from abc import ABC, abstractmethod
from enum import Enum, auto


def next_player(player_id):
    return 1-player_id

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

    #"""
    @property
    @abstractmethod 
    def player_turn(self):
        raise NotImplementedError
    #"""
  
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

#def fast_game(agent1, agent2, game_state) -> winner:
    
#    while not game_state.is_terminal():
#        if game_state.player_turn == Players.PLAYER1
