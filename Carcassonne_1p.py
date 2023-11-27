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
import Agents.siea_mcts2 as ea_mcts2
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
import multiprocessing as mp

#eamcts and sieamcts dont increase their iterations when they evolve

class ExperimentParser():
    def __init__(self, seed=0, games=1, runs=1, iterations=10, c=math.sqrt(2), meeples=7, rollouts=1, random_events=0, file_path="", lock = None, agent="mcts", iteration_snapshots=10):
        self.seed = seed
        self.games = games
        self.runs = runs
        self.iterations = iterations
        self.c = c
        self.meeples = meeples
        self.rollouts = rollouts
        self.random_events = random_events
        self.file_path = file_path
        self.lock = lock
        self.agent = agent
        self.iteration_snapshots = iteration_snapshots

#Parse arguments
parser = argparse.ArgumentParser(description='Run 1p Carcassone games')
"""
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
parser.add_argument('-a','--agent', type=str, default="mcts",help='mcts, eamcts, sieamcts, eamcts2')
#parser.add_argument('-p','--multiprocess', type=int, default=0,help='1=multiprocess, 0=not multiprocess')
"""


def run_experiment(parser):
#Declare parameters
    #parser = parser.parse_args()
    
    print("Experiment started with parameters:", parser.__dict__)

    #Create folder name
    this_file_path = parser.file_path
    this_file_path += "_" + parser.agent
    this_file_path += "_m" + str(parser.meeples)
    if parser.agent == "mcts": this_file_path += "_c" + str(parser.c)
    this_file_path += "_rand" + str(parser.random_events)
    this_file_path += "_it" + str(parser.iterations)

    #Store parser's parameters
    parser_df = pd.DataFrame(parser.__dict__, index=[0])
    lm.dump_data(file_path=this_file_path, data=parser_df, file_name="parameters.csv")
    es_lambda = 4
    es_fitness_iterations = 30
    es_generations = 20
    evolution_iterations = es_fitness_iterations*es_generations*es_lambda + es_fitness_iterations
    
    
    #Create game states
    game_state_seeds = [x for x in range(20,parser.games+20)]
    #game_state_seeds = [19,20]
    initial_game_states = [carc.CarcassonneState(name = "Carcassonne_less_1p",
                                                initial_tile_quantities=[1 for _ in range(24)],
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
    if parser.agent == "mcts":
        players = [mcts.MCTS_Player(max_iterations =parser.iterations,
                                        c=parser.c,
                                        logs=True,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "MCTS_c" + str(parser.c),
                                        rollouts=parser.rollouts)]
    elif parser.agent == "eamcts":
        players = [siea_mcts.SIEA_MCTS_Player(max_iterations = parser.iterations,# - evolution_iterations, 
                                          logs = True,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "EA_MCTS_its" + str(parser.iterations),# - evolution_iterations),
                                        rollouts=parser.rollouts,
                                        es_lambda = es_lambda,
                                        es_fitness_iterations = es_fitness_iterations,
                                        es_generations = es_generations,
                                          use_semantics=False)]
    elif parser.agent == "sieamcts":
        players = [siea_mcts.SIEA_MCTS_Player(max_iterations = parser.iterations,# - evolution_iterations, 
                                          logs = True,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "SIEA_MCTS_its" + str(parser.iterations),# - evolution_iterations),
                                          rollouts=parser.rollouts,
                                          es_lambda = es_lambda,
                                        es_fitness_iterations = es_fitness_iterations,
                                        es_generations = es_generations,
                                          use_semantics=True)]
    elif parser.agent == "eamcts2":
        players = [ea_mcts2.SIEA_MCTS_Player2(max_iterations = parser.iterations, 
                                          logs = True,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "EA_MCTS2_its" + str(parser.iterations),
                                        es_lambda = es_lambda,
                                        es_fitness_iterations = es_fitness_iterations,
                                        es_generations = es_generations,
                                          rollouts=parser.rollouts)]
    else: 
        raise ValueError("Agent not recognised:", parser.agent)

    #Create gameplayer

    for game_idx, game_state in enumerate(initial_game_states):
        print("Running game", str(game_idx+1), "of", str(parser.games))
        gameplayer = eu.GamePlayer(game_state, players)
        gameplayer.play_games(n_games=parser.runs, 
                            random_seed=parser.seed,
                            logs=True,
                            logs_path=os.path.join(this_file_path, "Game_" + str(game_idx+1)))

    #gameplayer.save_data(file_path=file_path)
    

#######Multiprocess version

if __name__ == "__main__":
    datetime_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_file_path = os.path.join("Outputs","Carcassonne_1p", "FINAL_Carcassonne_extra_games",  datetime_string)  ##DONT FORGET TO CHANGE THIS
    es_lambda = 4
    es_fitness_iterations = 30
    es_generations = 20
    evolution_iterations = es_fitness_iterations*es_generations*es_lambda + es_fitness_iterations
    parsers = []
    games = 30
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path, agent = "eamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path,agent = "eamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "eamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "eamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path, agent = "eamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path,agent = "eamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "eamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "eamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path,agent = "eamcts2")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path,agent = "eamcts2")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path,agent = "eamcts2")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path,agent = "eamcts2")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000+evolution_iterations, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path,agent = "eamcts2"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000+evolution_iterations, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path,agent = "eamcts2"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000+evolution_iterations, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path,agent = "eamcts2"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000+evolution_iterations, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path,agent = "eamcts2"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path, agent = "sieamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path, agent = "sieamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "sieamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "sieamcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path, agent = "sieamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path,agent = "sieamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "sieamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000-evolution_iterations, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "sieamcts"))
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=0.5, meeples=1, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=0.5, meeples=3, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=0.5, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=0.5, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts")) #ok
    parsers.append(ExperimentParser(seed=0, games=games, runs=1, iterations=5000, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts")) #ok
    

    batch_size = 6

    for i in range(math.ceil(len(parsers)/batch_size)):

        running_parser = []
        while len(running_parser) < batch_size and len(parsers) > 0:
            running_parser.append(parsers.pop(0))

        lock = mp.Lock()
        #print("Number of cpu : ", mp.cpu_count())
        assert mp.cpu_count() >= len(running_parser), "Not enough cpus to run all experiments"
        
        # Create and start processes
        processes = []
        for parser in running_parser:
            print("Starting process for parser", parser.__dict__)
            #p = mp.Process(target=run_experiment, args=(filenames[i],))
            p = mp.Process(target=run_experiment, args=(parser,))
            processes.append(p)
            p.start()
        
        # Wait for processes to finish
        for p in processes:
            p.join()