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
import scipy.stats
import importlib
import sys
import argparse

#Parse arguments
parser = argparse.ArgumentParser(description='Run 1p Carcassone games')
parser.add_argument('-r','--runs', type=int, default=1,
                    help='Number of runs for each game configuration. Greater than 1 will average results')
parser.add_argument('-g','--games', type=int, default=1,help='Number of game configurations')
parser.add_argument('-i','--iterations', type=int, default=10,
                    help='Number of mcts iterations')
parser.add_argument('-s','--seed', type=int, default=0,
                    help='Random seed')
parser.add_argument('-rand','--random_events', type=int, default=0,
                    help='Random events. 0 = no random events, 1 = random events')
parser.add_argument('-m','--meeples', type=int, default=7,
                    help='Carcassonne meeples')
parser.add_argument('-c','--c', type=float, default=math.sqrt(2),help='MCTS c parameter')
parser.add_argument('-roll','--rollouts', type=int, default=1,help='MCTS rollouts')

#Declare parameters
parser = parser.parse_args()
datetime_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
file_path = os.path.join("Outputs","Carcassonne_1p", datetime_string)
print("Experiment started with parameters:", parser.__dict__)

#Store parser's parameters
parser_df = pd.DataFrame(parser.__dict__, index=[0])
lm.dump_data(file_path=file_path, data=parser_df, file_name="parameters.csv")

#Create game states
game_state_seeds = [x for x in range(parser.games)]
initial_game_states = [carc.CarcassonneState(name = "Carcassonne_1p_set",
                                             set_tile_sequence= parser.random_events == 0,
                                             set_tile_sequence_seed=game_seed,
                                             initial_meeples = [parser.meeples, parser.meeples],
                                             players = 1) for game_seed in game_state_seeds]
for game_state in initial_game_states:
    game_state.set_initial_state()

#Set seeds
rd.seed(parser.seed)
np.random.seed(parser.seed)

#Create player
players = [mcts.MCTS_Player(max_iterations =parser.iterations,
                                c=parser.c,
                                logs=True,
                                name = "MCTS_c" + str(parser.c),
                                rollouts=parser.rollouts)]

#Create gameplayer

for game_idx, game_state in enumerate(initial_game_states):
    print("Running game", str(game_idx+1), "of", str(parser.games))
    gameplayer = eu.GamePlayer(game_state, players)
    gameplayer.play_games(n_games=parser.runs, 
                        random_seed=parser.seed,
                        logs=True,
                        logs_path=os.path.join(file_path, "Game_" + str(game_idx+1)))

#gameplayer.save_data(file_path=file_path)