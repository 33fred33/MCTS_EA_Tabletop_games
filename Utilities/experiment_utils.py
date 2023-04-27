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
    else: random_seed = rd.randint(0, 2**32)

    #Set logs
    df_logs = pd.DataFrame(columns=["player","player_turn", "chosen_action", "turn", "had_n_options", "time","random_seed"])

    #Play game
    while not gs.is_terminal:
        start_time = time.time()
        action = players[gs.player_turn].choose_action(gs)
        selection_time = time.time() - start_time

        #Update logs
        if logs:
            #ref: https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe
            df_logs = pd.concat([df_logs, pd.DataFrame([[
                str(players[gs.player_turn]),       #player
                gs.player_turn,                     #"player_index"
                str(action),                        #"chosen_action"
                gs.turn,                            #"turn":
                len(gs.available_actions),          #"had_n_options":
                selection_time,                      #"time":
                random_seed                         #"random_seed"
            ]], columns=df_logs.columns)], ignore_index=True)

        #Make action
        gs.make_action(action)
        
    return gs, df_logs