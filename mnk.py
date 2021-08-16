from typing import List, Tuple
import game
import numpy as np

class Action():#game.BaseAction):

    x : int
    y : int

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def duplicate(self):
        return Action(self.x, self.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))
    
    def __ne__(self, other):
        return not(self == other)

class GameState():#game.BaseGameState):

    m : int
    n : int
    k : int
    winner : int = None
    is_terminal : bool = False
    player_turn : int = 0
    available_actions : List[Action] = []

    def __init__(self, m=13, n=13, k=5):
        self.m = m
        self.n = n 
        self.k = k #line goal

    def set_initial_state(self):
        self.board = np.full((self.m, self.n), 2, dtype=np.uint)
        self.player_turn = 0
        for x in range(self.m):
            for y in range(self.n):
                self.available_actions.append(Action(x,y))
        self.is_terminal = False

    def make_action(self, action):
        self.board[action.x, action.y] = self.player_turn

        #Horizontal, vertical
        horizontal = self.board[action.x]
        vertical = self.board[:,action.y]
        offset = action.y-action.x
        antioffset = self.n - 1 - action.y - action.x
        diagonal_up = self.board.diagonal(offset)
        diagonal_down = np.fliplr(self.board).diagonal(antioffset)
        for seq in [horizontal, vertical, diagonal_up, diagonal_down]:
            count = 0
            for i in np.nditer(seq):
                if i == self.player_turn:
                    count += 1
                    if count >= self.k:
                        self.winner = self.player_turn
                        break
                else:
                    count = 0
            if self.winner is not None:
                self.is_terminal = True
                break

        self.available_actions.remove(action)
        if len(self.available_actions) == 0:
            self.is_terminal = True

        self.player_turn = 1 - self.player_turn
        
    def view_game_state(self):
        temp_board = np.full((self.m, self.n), " ")
        temp_board[self.board == 0] = "X"
        temp_board[self.board == 1] = "O"
        print(temp_board)

    def duplicate(self):
        the_duplicate = GameState(self.m, self.n, self.k)
        the_duplicate.winner = self.winner
        the_duplicate.player_turn = self.player_turn
        the_duplicate.available_actions = [a.duplicate() for a in self.available_actions]
        the_duplicate.board = np.matrix([[c for c in row] for row in self.board])
        return the_duplicate