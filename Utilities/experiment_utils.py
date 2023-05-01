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


def play_game(gs, players, random_seed = None, logs = False):
    "Reurns the terminal state of a game. Player is a list of agents."

    #Set random seed
    if random_seed is not None: rd.seed(random_seed)
    else: 
        random_seed = rd.randint(0, 2**32)
        rd.seed(random_seed)

    #Set logs
    acion_logs = pd.DataFrame(columns=["Player", "Chosen_action", "Time"])
    game_logs = pd.DataFrame()

    #Play game
    while not gs.is_terminal:
        start_time = time.time()
        action = players[gs.player_turn].choose_action(gs)
        selection_time = time.time() - start_time

        #Update logs
        if logs:
            #ref: https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe
            acion_logs = pd.concat([acion_logs, pd.DataFrame([[
                str(players[gs.player_turn]),       #player
                str(action),                        #"chosen_action"
                selection_time,                      #"time":
            ]], columns=acion_logs.columns)], ignore_index=True)
            game_logs = pd.concat([game_logs, gs.logs_data()], ignore_index=True)

        #Make action    
        gs.make_action(action)

    if logs:
        #Final logs by action
        final_logs_by_action = pd.concat([acion_logs, game_logs], axis=1)

        #Final logs by game
        final_logs_by_game_dict = {}
        for i, player in enumerate(players):
            final_logs_by_game_dict["Player_" + str(i)] = str(player)
        final_logs_by_game_dict["Random_seed"] = random_seed
        final_logs_by_game = pd.DataFrame(final_logs_by_game_dict, index=[0])
        final_logs_by_game = pd.concat([final_logs_by_game, gs.logs_data()], axis=1)
        
        return gs, {"by_game": final_logs_by_game, "by_action":final_logs_by_action}
    
    else:
        return gs, {}