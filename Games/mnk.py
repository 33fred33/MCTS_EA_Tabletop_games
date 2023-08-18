from typing import List, Tuple
import numpy as np
import Games.base_games as base_games
import pandas as pd
import random as rd

class Action():

    x : int
    y : int
    content : int

    def __init__(self, player_index, x, y):
        self.player_index = player_index
        self.x = x
        self.y = y

    def __str__(self):
        return f"(x{self.x},y{self.y},t{self.player_index})"
    
    def __members(self):
        return (self.x, self.y, self.player_index)
    
    def __eq__(self, other):
        return self.__members() == other.__members()

    def __hash__(self):
        return hash(self.__members())

class GameState(base_games.BaseGameState):

    name:str
    m : int
    n : int
    k : int
    winner : int = None
    is_terminal : bool = False
    player_turn : int = 0
    available_actions : List[Action] = []
    reward : List[float]
    direction_pairs = [["up","down"],["left","right"],["up_left","down_right"],["up_right","down_left"]]

    def __init__(self, m=13, n=13, k=5,
                 losing_reward=-1,
                 draw_reward=0,
                 winning_reward=1,
                 name=None):
        assert m >= 3, "m must be >= 3"
        assert n >= 3, "n must be >= 3"
        assert k >= 3 and (k <= m or k <= n), "k must be >= 3 and < m or n"
        if name is None:
            if k==5:
                self.name = "Gomoku"
            if m==n and k==3 and m==3:
                self.name = "TicTacToe"
            else:
                self.name = "mnk_m" + str(m) + "-n" + str(n) +"-k" + str(k)
        else: self.name = name
        self.m = m
        self.n = n 
        self.k = k
        self.losing_reward = losing_reward
        self.draw_reward = draw_reward
        self.winning_reward = winning_reward

    def set_initial_state(self):
        
        #Initialise critical variables
        self.turn = 1
        self.player_turn = 0
        self.is_terminal = False
        self.reward = [None,None]
        self.winner = None

        #Initialise board
        self.board = {}
        for x in range(self.m):
            for y in range(self.n):
                self.board[(x,y)] = None

        #Pre-calculate board connections
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

        #Initialise available actions
        self._available_actions = [{location:Action(player_index=player_index,x=location[0],y=location[1]) for location in self.board.keys()} for player_index in [0,1]]
        self._update_available_actions()

    def set_board(self, board_items, player_turn):
        self.set_initial_state()
        self.player_turn = player_turn
        for key,value in board_items.items():
            self.board[key] = value
        self._available_actions = [{location:Action(player_index=player_index,x=location[0],y=location[1]) for location in self.board.keys() if self.board[location] is None} for player_index in [0,1]]
        self._update_available_actions()

    def _update_available_actions(self):
        self.available_actions = [a for a in self._available_actions[self.player_turn].values()]
        #assert self.available_actions == [] and self.is_terminal, "No available actions after move and game is not terminal"

    def make_action(self, action):

        #Verify move validity
        assert self.is_terminal == False
        assert action.player_index == self.player_turn
        assert self.board[(action.x, action.y)] is None, "Invalid move:" + str(action)

        #Update board
        board_key = (action.x, action.y)
        self.board[board_key] = self.player_turn

        #Verify winner
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
                        break
                    current_board_key = self.board_connections[current_board_key][direction]

        if not self.is_terminal: 
            #Update available actions
            if len(self.available_actions) == 1:
                self._game_end(winner=None)
            else: 
                #Turn swap
                self.player_turn = 1 - self.player_turn
                del self._available_actions[0][board_key]
                del self._available_actions[1][board_key]
                self._update_available_actions()
                self.turn = self.turn + 1
        
        else:
            self._game_end(winner=self.player_turn)

    def _game_end(self, winner):
        self.is_terminal = True
        self.winner = winner
        self.available_actions = []
        if winner is None:
            self.reward = [self.draw_reward,self.draw_reward]
        else:
            self.reward[winner] = self.winning_reward
            self.reward[1-winner] = self.losing_reward

    def view_game_state(self):
        temp_board = np.full((self.m, self.n), " ")
        for x in range(self.m):
            for y in range(self.n):
                if self.board[(x,y)] is not None:
                    temp_board[x][self.n-y-1] = "X" if self.board[(x,y)]==0 else "O"
        print(temp_board.T)

    def duplicate(self):
        the_duplicate = GameState(m=self.m, 
                                  n=self.n,
                                  k=self.k, 
                                  losing_reward = self.losing_reward, 
                                  draw_reward = self.draw_reward, 
                                  winning_reward = self.winning_reward,
                                  name = self.name)
        the_duplicate.is_terminal = self.is_terminal
        the_duplicate.turn = self.turn
        the_duplicate.winner = self.winner
        the_duplicate.reward = [s for s in self.reward]
        the_duplicate.player_turn = self.player_turn
        the_duplicate._available_actions = [{c:v for c,v in self._available_actions[0].items()}, {c:v for c,v in self._available_actions[1].items()}]
        the_duplicate.available_actions = [a for a in self.available_actions]
        the_duplicate.board = {c:v for c,v in self.board.items()}
        the_duplicate.board_connections=self.board_connections
        return the_duplicate

    def feature_vector(self):
        fv = {k:v for k,v in self.board.items()}
        fv["player_turn"] = self.player_turn
        return fv

    def game_definition_data(self):
        data = {"m":self.m,
                "n":self.n,
                "k":self.k,
                "losing_reward":self.losing_reward,
                "draw_reward":self.draw_reward,
                "winning_reward":self.winning_reward,
                "name":self.name}
        return pd.DataFrame(data, index=[0])

    def logs_data(self):
        data = self.feature_vector()#{k:[v] for k,v in self.feature_vector().items()}
        for i, player_reward in enumerate(self.reward):
            data["Reward_p"+str(i)] = player_reward
        data["Turn"] = self.turn
        data["Winner"] = self.winner
        data["Is_terminal"] = self.is_terminal
        data["N_available_actions"] = len(self.available_actions)
        return pd.DataFrame(data, index=[0])

    def __repr__(self):
        return self.name + "_fv:" + str(self.feature_vector())