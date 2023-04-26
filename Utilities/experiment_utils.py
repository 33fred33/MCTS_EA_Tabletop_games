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
#import carcassonne as carc


def play_game(gs, players, logs = False):
    "Reurns the terminal state of a game. Player is a list of agents."
    df_logs = pd.DataFrame(columns=["player","player_turn", "chosen_action", "turn", "had_n_options", "time"])
    while not gs.is_terminal:
        start_time = time.time()
        action = players[gs.player_turn].choose_action(gs)
        selection_time = time.time() - start_time
        if logs:
            df_logs = pd.concat([df_logs, pd.DataFrame([[
                str(players[gs.player_turn]),       #player
                gs.player_turn,                     #"player_index"
                str(action),                        #"chosen_action"
                gs.turn,                            #"turn":
                len(gs.available_actions),          #"had_n_options":
                selection_time                      #"time":
            ]], columns=df_logs.columns)], ignore_index=True)
        gs.make_action(action)
    return gs, df_logs