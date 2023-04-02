from typing import List, Tuple
import numpy as np



class Action():#game.BaseAction):

    x : int
    y : int

    def __init__(self, index, x, y):
        self.index = index
        #self.content = content
        self.x = x
        self.y = y
    
    def duplicate(self):
        return Action(self.x, self.y)
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y# and self.content == other.content

    def __hash__(self):
        return hash((self.x, self.y))
    
    def __ne__(self, other):
        return not(self == other)

class GameState():

    m : int
    n : int
    k : int
    winner : int = None
    is_terminal : bool = False
    player_turn : int = 0
    available_actions : dict = {}

    def __init__(self, m=13, n=13, k=5, 
                 board={}, 
                 available_actions={}, 
                 player_turn=None, 
                 is_terminal=False, 
                 winner=None,
                 board_connections={}):
        self.m = m
        self.n = n 
        self.k = k
        self.board = board
        self.available_actions = available_actions
        self.player_turn = player_turn
        self.is_terminal = is_terminal
        self.winner = winner
        self.board_connections = board_connections
        self.direction_pairs = [["up","down"],["left","right"],["up_left","down_right"],["up_right","down_left"]]

    def set_initial_state(self):
        
        self.board = {}
        for x in range(self.m):
            for y in range(self.n):
                self.board[(x,y)] = None

        self.board_connections = {}
        for key in self.board.keys():
            x=key[0]
            y=key[1]
            self.board_connections[key] = {"up":(x, y+1),
                                       "left":(x-1, y),
                                       "down":(x,y-1),
                                       "right":(x+1,y),
                                       "up_left":(x-1,y+1),
                                       "up_right":(x+1,y+1),
                                       "down_left":(x-1,y-1),
                                       "down_right":(x+1,y-1)}
            if x == 0:
                for connection_key in self.board_connections[key].keys():
                    if "left" in connection_key:
                        self.board_connections[key][connection_key] = None
            elif x == self.m-1:
                for connection_key in self.board_connections[key].keys():
                    if "right" in connection_key:
                        self.board_connections[key][connection_key] = None
            if y == 0:
                for connection_key in self.board_connections[key].keys():
                    if "down" in connection_key:
                        self.board_connections[key][connection_key] = None
            elif y == self.n-1:
                for connection_key in self.board_connections[key].keys():
                    if "up" in connection_key:
                        self.board_connections[key][connection_key] = None

        self.player_turn = 0
        self.is_terminal = False

        action_counter = 0
        for x in range(self.m):
            for y in range(self.n):
                self.available_actions[action_counter] = Action(index=action_counter,
                                                                #content=self.player_turn,
                                                                x=x,y=y)
                action_counter += 1

    def make_action(self, action):
        board_key = (action.x, action.y)
        self.board[board_key] = self.player_turn

        #Verify end
        for direction_pair in self.direction_pairs:
            line_count = 1
            for direction in direction_pair:
                current_board_key = board_key
                while self.board_connections[current_board_key][direction] is not None:
                    if self.board[self.board_connections[current_board_key][direction]] != self.player_turn:
                        break
                    line_count += 1
                    if line_count==self.k:
                        self.is_terminal = True
                        self.winner = self.player_turn
                        break
                    current_board_key = self.board_connections[current_board_key][direction]

        del self.available_actions[action.index]
        if len(self.available_actions) == 0:
            self.is_terminal = True
            self.winner = None

        #Turn swap
        self.player_turn = 1 - self.player_turn
        
    def view_game_state(self):
        temp_board = np.full((self.m, self.n), " ")
        for x in range(self.m):
            for y in range(self.n):
                if self.board[(x,y)] is not None:
                    temp_board[x][y] = "X" if self.board[(x,y)]==0 else "O"
        print(temp_board)

    def duplicate(self):
        the_duplicate = GameState(self.m, self.n, self.k, 
                                  winner = self.winner, 
                                  player_turn = self.player_turn, 
                                  available_actions = {a:v for a,v in self.available_actions.items()}, 
                                  board = {c:v for c,v in self.board.items()},
                                  board_connections=self.board_connections)
        return the_duplicate
    
    def __repr__(self):
        return "m"+str(self.m)+"n"+str(self.n) +"k"+str(self.k) +"terminal"+str(self.is_terminal) +"winner"+str(self.winner) + "player_turn" +str(self.player_turn) + "available_actions" + str(len(self.available_actions))