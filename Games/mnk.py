from typing import List, Tuple
import numpy as np
import Games.base_games as base_games


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
    available_actions = []
    direction_pairs = [["up","down"],["left","right"],["up_left","down_right"],["up_right","down_left"]]

    def __init__(self, m=13, n=13, k=5,
                 turn=None,
                 board={}, 
                 _available_actions=[{},{}],
                 available_actions={}, 
                 player_turn=None, 
                 is_terminal=False, 
                 winner=None,
                 board_connections={}):
        self.m = m
        self.n = n 
        self.k = k
        self.board = board

        #supporting variable to speed up the games
        self._available_actions = _available_actions

        self.available_actions = available_actions
        self.player_turn = player_turn
        self.is_terminal = is_terminal
        self.winner = winner
        self.board_connections = board_connections
        self.turn = turn

    def set_initial_state(self):
        
        #Initialise critical variables
        self.turn = 1
        self.player_turn = 0
        self.is_terminal = False

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
                        self.winner = self.player_turn
                        break
                    current_board_key = self.board_connections[current_board_key][direction]

        if not self.is_terminal: 
            #Update available actions
            if len(self.available_actions) == 1:
                self.is_terminal = True
                self.winner = None
                self.available_actions = []
            else: 
                #Turn swap
                self.player_turn = 1 - self.player_turn
                del self._available_actions[0][board_key]
                del self._available_actions[1][board_key]
                self._update_available_actions()
                self.turn = self.turn + 1
        
        else:
            self.available_actions = []
        
    def view_game_state(self):
        temp_board = np.full((self.m, self.n), " ")
        for x in range(self.m):
            for y in range(self.n):
                if self.board[(x,y)] is not None:
                    temp_board[x][y] = "X" if self.board[(x,y)]==0 else "O"
        print(temp_board)

    def duplicate(self):
        the_duplicate = GameState(self.m, self.n, self.k, 
                                  turn = self.turn,
                                  winner = self.winner, 
                                  player_turn = self.player_turn, 
                                  _available_actions = [{c:v for c,v in self._available_actions[0].items()}, {c:v for c,v in self._available_actions[1].items()}],
                                  available_actions = [a for a in self.available_actions], 
                                  board = {c:v for c,v in self.board.items()},
                                  board_connections=self.board_connections)
        return the_duplicate
    
    def __repr__(self):
        return "m"+str(self.m)+"n"+str(self.n) +"k"+str(self.k) +"terminal"+str(self.is_terminal) +"winner"+str(self.winner) + "player_turn" +str(self.player_turn) + "available_actions" + str(len(self.available_actions))