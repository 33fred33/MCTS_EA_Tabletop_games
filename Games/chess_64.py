from typing import List, Tuple
import numpy as np
import Games.base_games as base_games
import pandas as pd
import chess

class Action():

    def __init__(self, move, player_turn):
        self.move = move
        self.player_turn = player_turn

    def __str__(self):
        pt_str = "White" if self.player_turn == 0 else "Black"
        return  pt_str + ":" + str(self.move)
    
    def __eq__(self, other):
        return str(self.move) == str(other.move) and self.player_turn == other.player_turn

    def __hash__(self):
        return hash(str(self.move) + str(self.player_turn))
    
class GameState(base_games.BaseGameState):
    winner : int = None
    is_terminal : bool = False
    player_turn : int = 0
    available_actions : List[Action] = []
    reward : List[float]
    def __init__(self, losing_reward=-1,
                 draw_reward=0,
                 winning_reward=1, name=None) -> None:
        if name is None: self.name = "Chess"
        else: self.name = name
        self.losing_reward = losing_reward
        self.draw_reward = draw_reward
        self.winning_reward = winning_reward
    
    def set_initial_state(self) -> None:
        self.turn = 1
        self.player_turn = 0
        self.is_terminal = False
        self.reward = [None, None]
        self.winner = None
        self.board = chess.Board()

        self.available_actions = [Action(move, player_turn=self.player_turn) for move in list(self.board.legal_moves)]
    
    def set_state_from_FEN(self, FEN):
        split_FEN = FEN.split(" ")
        if split_FEN[1] == "w": 
            self.player_turn = 0
        else: 
            self.player_turn = 1
        self.turn = 1
        self.is_terminal = False
        self.reward = [None, None]
        self.winner = None
        self.board = chess.Board(FEN)

        self.available_actions = [Action(move, player_turn=self.player_turn) for move in list(self.board.legal_moves)]

    def set_puzzle_lichess_db(self, puzzle_row):
        self.set_state_from_FEN(puzzle_row["FEN"])

        #The first move in the puzzle is the last move played from the FEN
        move_sequence = puzzle_row["Moves"].split(" ")
        self.make_action(Action(self.board.parse_uci(move_sequence[0]), player_turn=self.player_turn))

    def make_action(self, action):
        assert action.player_turn == self.player_turn, "Wrong player turn"
        self.board.push(action.move)
        #self.available_actions = [Action(move) for move in list(self.board.legal_moves)]
        if self.board.is_game_over():
            if self.board.outcome().winner is None:
                self._game_end(None)
            elif self.board.outcome().winner: self._game_end(0)
            else: self._game_end(1)
            
        else:
            self.player_turn = 1 - self.player_turn
            self.turn += 1
            self.available_actions = [Action(move, player_turn=self.player_turn) for move in list(self.board.legal_moves)]

    def _game_end(self, winner):
        self.is_terminal = True
        self.winner = winner
        if winner is None:
            self.reward = [self.draw_reward,self.draw_reward]
        else:
            self.reward[winner] = self.winning_reward
            self.reward[1-winner] = self.losing_reward

    def view_game_state(self):
        return print(self.board)

    def duplicate(self):
        the_duplicate = GameState(losing_reward = self.losing_reward, 
                                  draw_reward = self.draw_reward, 
                                  winning_reward = self.winning_reward,
                                  name = self.name)
        the_duplicate.is_terminal = self.is_terminal
        the_duplicate.turn = self.turn #self.board.ply()
        the_duplicate.winner = self.winner
        the_duplicate.reward = [s for s in self.reward]
        the_duplicate.player_turn = self.player_turn
        the_duplicate.available_actions = [Action(move, player_turn=self.player_turn) for move in list(self.board.legal_moves)]
        the_duplicate.board = self.board.copy()
        return the_duplicate

    def feature_vector(self):
        #state.lan(rd.choice(list(state.legal_moves))) returns the square from and to
        #state.peek returns last move
        fv = {k:str(v) for k,v in self.board.piece_map().items()}
        for k in range(64):
            if k not in fv.keys():
                fv[k] = None
        fv["player_turn"] = self.player_turn
        return fv

    def game_definition_data(self):
        data = {"losing_reward":self.losing_reward,
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