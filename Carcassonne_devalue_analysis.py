import cProfile
import os
import statistics as st
import plotly.graph_objects as go
import random as rd
import time
import pandas as pd
import math
from collections import defaultdict
import numpy as np
import itertools as it
import Utilities.experiment_utils as eu
import unit_test as ut
import Games.mnk as mnk
import Games.Carcassonne.Carcassonne as carc
import Games.carcassonne_oldtry as csn_old
#import Games.carcassonne_older as csn
import Agents.random as arand
import Agents.vanilla_mcts as mcts
import Agents.siea_mcts as siea_mcts
import Agents.mcts_rave as rave_mcts
import Agents.one_step_lookahead as osla
import Agents.mcts_solver as mcts_solver
import Games.function_optimisation as fo
import Utilities.logs_management as lm
import Games.chess_64 as chess_64
import chess as chess
import chess.svg
import matplotlib.pyplot as plt
import shutil
import itertools
from IPython.display import display
import ast
from IPython.display import Image
import datetime
import multiprocessing as mp
from plotly.subplots import make_subplots

#cProfile.run("wins =  random_games(10000, base_gs)")
#ut.run()

import scipy.stats


#devalue analysis
"""
if second turn and tile=13, the score of p1 is up by 4 given that it closed a city with meeple (was needed to find the correct action)
if tile=0, the score is constant

"""

#paramters:
full_tiles = True
deterministic = False
game_seed = 0
meeples_options = [7]#[1,2,3]
players = 2
devalue_games = 100
second_turn = True
enforced_fist_turn_tile = 0 #13 for closing a city, 0 for 

for meeples in meeples_options:
    exp_name = "meeples " + str(meeples) + "_games " + str(devalue_games) + "_seed " + str(game_seed) + "_fulltiles " + str(full_tiles) + "_deterministic " + str(deterministic)
    if second_turn:
        exp_name = exp_name + "2nd_turn_after_tile" + str(enforced_fist_turn_tile)
    output_path = os.path.join("Outputs","Carcassonne_analysis", exp_name)

    #calculations
    if full_tiles:
        initial_tile_quantities = [1,3,1,1,2,3,2,2,2,3,1,3,2,5,3,2,4,3,3,4,4,9,8,1]
    else:
        initial_tile_quantities = [1 for _ in range(24)]

    #Initialise carcassonne game
    game_state = carc.CarcassonneState(name = "Carcassonne" + ("_fulltiles" if full_tiles else "_lesstiles") + ("_deterministic" if deterministic else "_stochastic") + ("_seed" + str(game_seed) if deterministic else ""),
                                                    initial_tile_quantities = initial_tile_quantities,
                                                    set_tile_sequence= deterministic,
                                                    set_tile_sequence_seed=game_seed,
                                                    initial_meeples = [meeples, meeples],
                                                    players = players)
    game_state.set_initial_state()

    if second_turn:
        game_state.sample_random_event(event_type = "Draw random tile", enforced_outcome = enforced_fist_turn_tile)

        if enforced_fist_turn_tile == 13:
            #find and play action that closes the initial city
            for action in game_state.available_actions:
                fclone_state = game_state.duplicate()
                fclone_state.make_action(action)
                if fclone_state.Scores[0] == 4:
                    print("winning action", action)
                    break
        else:
            for action in game_state.available_actions:
                fclone_state = game_state.duplicate()
                fclone_state.make_action(action)
                if fclone_state.Scores[2] == 0:
                    print("winning action", action)
                    break
        
        game_state = fclone_state    

    #Initialise agent: 
    agent = arand.RandomPlayer()

    for tile_index, tile_quantity in enumerate(game_state.TileQuantities):
        if tile_quantity > 0:
            if second_turn:
                if enforced_fist_turn_tile == 13:
                    if tile_index == 0:
                        continue
            game_state.sample_random_event(event_type = "Draw random tile", enforced_outcome = tile_index)
            for action_index, available_action in enumerate(game_state.available_actions):
                if "Meeple Location" in str(available_action):
                    meeple_type = str(available_action).split("Meeple Location: ")[1].split(",")[0]
                else:
                    meeple_type = "Non_Meeple"
                print(meeple_type, " in action", available_action, ". Playing game")


                clone_state = game_state.duplicate()
                clone_state.name = clone_state.name + "_fromaction_" + str(available_action)
                clone_state.make_action(available_action)

                gameplayer = eu.GamePlayer(clone_state, [agent for _ in range(players)])
                gameplayer.play_games(n_games=devalue_games, 
                                    random_seed=game_seed,
                                    logs=True,
                                    logs_path=os.path.join(output_path, "Tile_"+str(tile_index), "Action_index_" + str(action_index), meeple_type))
                

