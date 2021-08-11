from typing import List, Tuple
import game
import numpy as np

class Action(game.BaseAction):

    x : int
    y : int

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def duplicate(self):
        return Action(self.x, self.y)

class GameState(game.BaseGameState):

    m : int = 13
    n : int = 13
    k : int = 5
    winner : int = None
    player_turn : int = 0
    available_actions : List[Action] = []

    def __init__(self, m, n, k):
        self.m = m
        self.n = n 
        self.k = k #line goal

    def set_initial_state(self):
        self.board = np.full((self.m, self.n), 2, dtype=np.uint)
        self.player_turn = 1

    def make_action(self, action):
        self.board[action.x, action.y] = self.player_turn

        #Horizontal, vertical
        horizontal = self.board[action.x]
        vertical = self.board[:,action.y]
        diagonal_up = self.board.diagonal(action.y-action.x)
        diagonal_down = np.flipud(self.board).diagonal(action.y-action.x)
        for seq in [horizontal, vertical, diagonal_up, diagonal_down]:
            count = 0
            for i in seq:
                if i == self.player_turn:
                    count += 1
                    if count == self.k:
                        self.winner = self.player_turn
                        break
                else:
                    count = 0
            if self.winner is not None: 
                break
        
        #Diagonal
        

        self.player_turn = 1 - self.player_turn
        
    def is_terminal(self):
        return self.winner == None

    def duplicate(self):
        the_duplicate = GameState(self.m, self.n, self.k)
        the_duplicate.winner = self.winner
        the_duplicate.player_turn = self.player_turn
        the_duplicate.available_actions = [a.duplicate() for a in self.available_actions]
        the_duplicate.board = np.matrix([[c for c in row] for row in self.board])