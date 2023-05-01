from typing import List, Tuple
import numpy as np
import Games.base_games as base_games
import pandas as pd

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

    m : int
    n : int
    k : int
    winner : int = None
    is_terminal : bool = False
    player_turn : int = 0
    available_actions : List[Action] = []
    score : List[float]
    direction_pairs = [["up","down"],["left","right"],["up_left","down_right"],["up_right","down_left"]]

    def __init__(self, m=13, n=13, k=5,
                 losing_score=-1,
                 draw_score=0,
                 winning_score=1):
        assert m >= 3, "m must be >= 3"
        assert n >= 3, "n must be >= 3"
        assert k >= 3 and (k <= m or k <= n), "k must be >= 3 and < m or n"
        self.m = m
        self.n = n 
        self.k = k
        self.losing_score = losing_score
        self.draw_score = draw_score
        self.winning_score = winning_score

    def set_initial_state(self):
        
        #Initialise critical variables
        self.turn = 1
        self.player_turn = 0
        self.is_terminal = False
        self.score = [None,None]
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

    def _update_available_actions(self):
        self.available_actions = [a for a in self._available_actions[self.player_turn].values()]

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
            self.score = [self.draw_score,self.draw_score]
        else:
            self.score[winner] = self.winning_score
            self.score[1-winner] = self.losing_score

    def view_game_state(self):
        temp_board = np.full((self.m, self.n), " ")
        for x in range(self.m):
            for y in range(self.n):
                if self.board[(x,y)] is not None:
                    temp_board[x][y] = "X" if self.board[(x,y)]==0 else "O"
        print(temp_board)

    def duplicate(self):
        the_duplicate = GameState(m=self.m, 
                                  n=self.n,
                                  k=self.k, 
                                  losing_score = self.losing_score, 
                                  draw_score = self.draw_score, 
                                  winning_score = self.winning_score)
        the_duplicate.turn = self.turn
        the_duplicate.winner = self.winner
        the_duplicate.score = [s for s in self.score]
        the_duplicate.player_turn = self.player_turn
        the_duplicate._available_actions = [{c:v for c,v in self._available_actions[0].items()}, {c:v for c,v in self._available_actions[1].items()}]
        the_duplicate.available_actions = [a for a in self.available_actions]
        the_duplicate.board = {c:v for c,v in self.board.items()}
        the_duplicate.board_connections=self.board_connections
        return the_duplicate

    def feature_vector(self):
        fv = {k:v for k,v in self.board.items()}
        return fv

    def logs_data(self):
        data = self.feature_vector()#{k:[v] for k,v in self.feature_vector().items()}
        for i, player_score in enumerate(self.score):
            data["Score_p"+str(i)] = player_score
        data["Turn"] = self.turn
        data["Winner"] = self.winner
        data["N_available_actions"] = len(self.available_actions)
        return pd.DataFrame(data, index=[0])

    def __repr__(self):
        return str(self.feature_vector())