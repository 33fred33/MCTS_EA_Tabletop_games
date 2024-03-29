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
import Agents.minimax as minimax
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

def heuristic_virtual_score(state, player): #default for carcassonne
    if state.is_terminal:
        to_return = state.reward[player]
    else:
        to_return = state.Scores[player+2]/state.max_possible_score
    assert to_return is not None, "Heuristic function returned None in state: " + str(state) + " and player: " + str(player)
    return to_return

def heuristic_score(state, player): #default for carcassonne
    if state.is_terminal:
        to_return = state.reward[player]
    else:
        to_return = state.Scores[player]/state.max_possible_score
    assert to_return is not None, "Heuristic function returned None in state: " + str(state) + " and player: " + str(player)
    return to_return

class ExperimentParser():
    def __init__(self, seed=0, games=1, runs=1, iterations=10, c=math.sqrt(2), meeples=7, rollouts=1, random_events=0, file_path="", lock = None, agent="mcts", iteration_snapshots=10, minimax_max_depth=1, minimax_star_L=0, minimax_star_U=1, minimax_probing_factor=1, heuristic_function=heuristic_score, move_ordering_function=None, agent_name=None, move_time = np.inf):
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
        self.minimax_max_depth = minimax_max_depth
        self.minimax_star_L = minimax_star_L
        self.minimax_star_U = minimax_star_U
        self.minimax_probing_factor = minimax_probing_factor
        self.heuristic_function = heuristic_function
        self.move_ordering_function = move_ordering_function
        self.agent_name = agent_name
        self.move_time = move_time

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
    #added_name = parser.agent if parser.agent_name is None else parser.agent_name
    this_file_path += "_" + parser.agent
    this_file_path += "_m" + str(parser.meeples)
    if parser.agent == "mcts": this_file_path += "_c" + str(parser.c)
    if parser.agent == "minimax":
        this_file_path += "_d" + str(parser.minimax_max_depth)
        this_file_path += "_L" + str(parser.minimax_star_L)
        this_file_path += "_U" + str(parser.minimax_star_U)
        this_file_path += "_pf" + str(parser.minimax_probing_factor)
        this_file_path += "_hf" + parser.heuristic_function.__name__
    this_file_path += "_rand" + str(parser.random_events)
    this_file_path += "_it" + str(parser.iterations)
    this_file_path += "_mt" + str(parser.move_time)

    


    #Store parser's parameters
    parser_df = pd.DataFrame(parser.__dict__, index=[0])
    lm.dump_data(file_path=this_file_path, data=parser_df, file_name="parameters.csv")
    es_lambda = 4
    es_fitness_iterations = 30
    es_generations = 20
    sem_L = 0.1 #for rewards between 0 and 1
    sem_U = 0.5 #for rewards between 0 and 1
    evolution_iterations = es_fitness_iterations*es_generations*es_lambda + es_fitness_iterations
    
    
    #Create game states
    game_state_seeds = [x for x in range(parser.games)]
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
                                    max_time = parser.move_time,
                                        c=parser.c,
                                        logs=True,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "MCTS_c" + str(parser.c) if parser.agent_name is None else parser.agent_name,
                                        rollouts=parser.rollouts)]
    elif parser.agent == "eamcts":
        players = [siea_mcts.SIEA_MCTS_Player(max_iterations = parser.iterations,# - evolution_iterations, 
                                              max_time = parser.move_time,
                                          logs = True,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "EA_MCTS" if parser.agent_name is None else parser.agent_name,
                                        rollouts=parser.rollouts,
                                        es_lambda = es_lambda,
                                        es_fitness_iterations = es_fitness_iterations,
                                        es_generations = es_generations,
                                          use_semantics=False)]
    elif parser.agent == "sieamcts":
        players = [siea_mcts.SIEA_MCTS_Player(max_iterations = parser.iterations,# - evolution_iterations, 
                                          logs = True,
                                          max_time = parser.move_time,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "SIEA_MCTS" if parser.agent_name is None else parser.agent_name,
                                          rollouts=parser.rollouts,
                                          es_lambda = es_lambda,
                                        es_fitness_iterations = es_fitness_iterations,
                                        es_generations = es_generations,
                                        es_semantics_l = sem_L,
                                        es_semantics_u = sem_U,
                                          use_semantics=True)]
    elif parser.agent == "eamcts2":
        players = [ea_mcts2.SIEA_MCTS_Player2(max_iterations = parser.iterations, 
                                          logs = True,
                                            max_time = parser.move_time,
                                        logs_every_iterations = int(parser.iterations/parser.iteration_snapshots),
                                        name = "EA_MCTS2" if parser.agent_name is None else parser.agent_name,
                                        es_lambda = es_lambda,
                                        es_fitness_iterations = es_fitness_iterations,
                                        es_generations = es_generations,
                                          rollouts=parser.rollouts)]
    elif parser.agent == "minimax":
        players = [minimax.Minimax(max_depth = parser.minimax_max_depth,
                                   name = "Minimax" if parser.agent_name is None else parser.agent_name,
                                   star_L=parser.minimax_star_L,
                                   star_U=parser.minimax_star_U,
                                   probing_factor = parser.minimax_probing_factor,
                                   heuristic_function = parser.heuristic_function,
                                   move_ordering_function = parser.move_ordering_function,
                                   logs=True,)]
    elif parser.agent == "random":
        players = [arand.RandomPlayer(name="Random" if parser.agent_name is None else parser.agent_name, 
                                      logs=True)]
    else: 
        raise ValueError("Agent not recognised:", parser.agent)

    #Create gameplayer

    for game_idx, game_state in enumerate(initial_game_states):
        print("Running game", str(game_idx+1), "of", str(parser.games))
        gameplayer = eu.GamePlayer(game_state, players)
        gameplayer.play_games(n_games=parser.runs, 
                            random_seed=parser.seed + game_idx,
                            logs=True,
                            logs_path=os.path.join(this_file_path, "Game_" + str(game_idx+1)))

    #gameplayer.save_data(file_path=file_path)
    

#######Multiprocess version

if __name__ == "__main__":
    datetime_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_file_path = os.path.join("Outputs","Carcassonne_1p", "FINAL_15_SEC",  datetime_string)  ##DONT FORGET TO CHANGE THIS
    es_lambda = 4
    es_fitness_iterations = 30
    es_generations = 20
    evolution_iterations = es_fitness_iterations*es_generations*es_lambda + es_fitness_iterations
    parsers = []
    games = 30
    rd_seed = 10
    iterations = 1000000
    move_time = 15
    #"""
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=1, rollouts=1, random_events=1, file_path=default_file_path, agent = "eamcts", move_time=move_time, agent_name="EA_MCTS_15")) #ok
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=3, rollouts=1, random_events=1, file_path=default_file_path,agent = "eamcts", move_time=move_time, agent_name="EA_MCTS_15")) #ok
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "eamcts" , move_time=move_time, agent_name="EA_MCTS_15")) #ok
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "eamcts" , move_time=move_time, agent_name="EA_MCTS_15")) #ok
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=1, rollouts=1, random_events=1, file_path=default_file_path, agent = "sieamcts", move_time=move_time, agent_name="SIEA_MCTS_15")) #ok
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=3, rollouts=1, random_events=1, file_path=default_file_path, agent = "sieamcts", move_time=move_time, agent_name="SIEA_MCTS_15")) #ok
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "sieamcts", move_time=move_time, agent_name="SIEA_MCTS_15")) #ok
    parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "sieamcts", move_time=move_time, agent_name="SIEA_MCTS_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=0.5, meeples=1, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c0.5_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=0.5, meeples=3, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c0.5_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=0.5, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c0.5_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=0.5, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c0.5_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=1, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c1.4142135623730951_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=3, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c1.4142135623730951_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c1.4142135623730951_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c1.4142135623730951_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=3, meeples=1, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c3_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=3, meeples=3, rollouts=1, random_events=1, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c3_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=3, meeples=1, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c3_15")) #ok
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, iterations=iterations, c=3, meeples=3, rollouts=1, random_events=0, file_path=default_file_path ,agent = "mcts", move_time=move_time, agent_name="MCTS_c3_15")) #ok
    """
    for ini_meeples in [1,3]:
        for rand_events in [1, 0]:
            parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, agent="minimax", minimax_max_depth=1, minimax_star_L=0, minimax_star_U=1, minimax_probing_factor=1, heuristic_function=heuristic_score, file_path=default_file_path, agent_name = "Minimax_1_s", meeples=ini_meeples, random_events=rand_events)) #ok
            parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, agent="minimax", minimax_max_depth=2, minimax_star_L=0, minimax_star_U=1, minimax_probing_factor=1, heuristic_function=heuristic_score, file_path=default_file_path, agent_name = "Minimax_2_s", meeples=ini_meeples, random_events=rand_events)) #ok
            parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, agent="minimax", minimax_max_depth=1, minimax_star_L=0, minimax_star_U=1, minimax_probing_factor=1, heuristic_function=heuristic_virtual_score,file_path=default_file_path, agent_name = "Minimax_1_vs", meeples=ini_meeples, random_events=rand_events)) #ok
            parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, agent="minimax", minimax_max_depth=2, minimax_star_L=0, minimax_star_U=1, minimax_probing_factor=1, heuristic_function=heuristic_virtual_score,file_path=default_file_path, agent_name = "Minimax_2_vs", meeples=ini_meeples, random_events=rand_events)) #ok
            parsers.append(ExperimentParser(seed=rd_seed, games=games, runs=1, agent="random", file_path=default_file_path, meeples=ini_meeples, random_events=rand_events)) #ok
    """

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