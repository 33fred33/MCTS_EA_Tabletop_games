import cProfile
import random as rd
import time
from collections import defaultdict
import numpy as np
from itertools import cycle
import pandas as pd

import Games.mnk as mnk
import Agents.random as arand
import Agents.vanilla_mcts as mcts 


class GamePlayer():
    def __init__(self, game_state, players) -> None:
        self.game_state = game_state
        self.players = players
        self.games_count = 0
        self.logs_by_game = pd.DataFrame()
        self.logs_by_action = pd.DataFrame()
        self.win_count = {i:0 for i in range(len(players))}
        self.win_count["Draw"] = 0
        
    def play_game(self, random_seed = None, logs = True):
        "Plays a game. If logs is true, adds data to the class logs"

        gs = self.game_state.duplicate()

        #Set random seed
        if random_seed is not None: rd.seed(random_seed)
        else: 
            random_seed = rd.randint(0, 2**32)
            rd.seed(random_seed)

        #Set logs
        action_logs = pd.DataFrame(columns=["Player", "Chosen_action", "Time", "Game_index"])
        game_logs = pd.DataFrame()

        #Play game
        while not gs.is_terminal:
            start_time = time.time()
            action = self.players[gs.player_turn].choose_action(gs)
            selection_time = time.time() - start_time

            #Update logs
            if logs:
                #ref: https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe
                action_logs = pd.concat([action_logs, pd.DataFrame([[
                    str(self.players[gs.player_turn]),       #player
                    str(action),                        #"chosen_action"
                    selection_time,                      #"time"
                    self.games_count                   #"game_index"
                ]], columns=action_logs.columns)], ignore_index=True)
                game_logs = pd.concat([game_logs, gs.logs_data()], ignore_index=True)

            #Make action    
            gs.make_action(action)

        if logs:
            #Final logs by action
            final_logs_by_action = pd.concat([action_logs, game_logs], axis=1)

            #Final logs by game
            final_logs_by_game_dict = {}
            for i, player in enumerate(self.players):
                final_logs_by_game_dict["Player_" + str(i)] = str(player)
            final_logs_by_game_dict["Random_seed"] = random_seed
            final_logs_by_game_dict["Game_index"] = self.games_count
            final_logs_by_game = pd.DataFrame(final_logs_by_game_dict, index=[0])
            final_logs_by_game = pd.concat([final_logs_by_game, gs.logs_data()], axis=1)
            
            #Update class logs
            self._update_logs_by_game(final_logs_by_game)
            self._update_logs_by_action(final_logs_by_action)
            self._update_win_count(gs.winner)
            self.games_count += 1
        return gs

    def _update_win_count(self, winner):
        if winner is None: self.win_count["Draw"] += 1
        else:
            self.win_count[winner] += 1

    def _update_logs_by_game(self, logs):
        self.logs_by_game = pd.concat([self.logs_by_game, logs], ignore_index=True)

    def _update_logs_by_action(self, logs):
        self.logs_by_action = pd.concat([self.logs_by_action, logs], ignore_index=True)

    def play_games(self, n_games, random_seed = None, logs = True):
        "Plays n_games games"
        if random_seed is not None: rd.seed(random_seed)
        else: rd.seed(rd.randint(0, 2**32))
        seeds = rd.sample(range(0, 2**32), n_games)
        for i in range(n_games):
            self.play_game(random_seed = seeds[i], logs = logs)