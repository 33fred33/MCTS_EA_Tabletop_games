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
    def __init__(self, file_folder, file_path="", lock=None, seed=0, games=1, runs=1, iterations1=10, iterations2 = 10, max_time1 = np.inf, max_time2 = np.inf, name1=None, name2=None, logs1=True, logs2=True, c1=math.sqrt(2), c2=math.sqrt(2), rollouts1=1, rollouts2=1, agent1="mcts", agent2="random", minimax_max_depth1=1, minimax_max_depth2=1, minimax_star_L1=0, minimax_star_L2=0, minimax_star_U1=1, minimax_star_U2=1, minimax_probing_factor1=1, minimax_probing_factor2=1, heuristic_function1=heuristic_score, heuristic_function2=heuristic_score, move_ordering_function1=None, move_ordering_function2=None):
        #agent1 parameters
        self.agents = [agent1,agent2]
        self.iterations = [iterations1,iterations2]
        self.max_times = [max_time1,max_time2]
        self.c = [c1,c2]
        self.rollouts = [rollouts1,rollouts2]
        self.names = [name1,name2]
        self.agent_logs = [logs1,logs2]
        self.minimax_max_depth = [minimax_max_depth1, minimax_max_depth2]
        self.minimax_star_L = [minimax_star_L1, minimax_star_L2]
        self.minimax_star_U = [minimax_star_U1, minimax_star_U2]
        self.minimax_probing_factor = [minimax_probing_factor1, minimax_probing_factor2]
        self.heuristic_function = [heuristic_function1, heuristic_function2]
        self.move_ordering_function = [move_ordering_function1, move_ordering_function2]
        #gameplayer parameters
        self.seed = seed
        self.games = games
        self.runs = runs
        #output parameters
        self.file_path = file_path
        self.file_folder = file_folder
        self.lock = lock
        
        

def run_experiment(parser):
    #Runs a match with the defined agents
    
    print("Experiment started with parameters:", parser.__dict__)
    #Create folder name
    this_file_path = os.path.join(parser.file_path, parser.file_folder)

    #Store parser's parameters
    parser_df = pd.DataFrame({str(k):str(v) for k,v in parser.__dict__.items()}, index=[0])
    lm.dump_data(file_path=this_file_path, data=parser_df, file_name="parameters.csv")

    #default parameters
    es_lambda = 4
    es_fitness_iterations = 30
    es_generations = 20
    sem_L = 0.1 #for rewards between 0 and 1
    sem_U = 0.5 #for rewards between 0 and 1
    
    #Create game states
    game_state_seeds = [x for x in range(parser.games)]
    initial_game_states = [carc.CarcassonneState() for game_seed in game_state_seeds]
    for game_state in initial_game_states:
        game_state.set_initial_state()

    #Set seeds
    rd.seed(parser.seed)
    np.random.seed(parser.seed)

    #Create player
    players = [None, None]
    for agent_index, agent_name in enumerate(parser.agents):
        if agent_name == "mcts":
            players[agent_index] = mcts.MCTS_Player(max_iterations = parser.iterations[agent_index],
                                        max_time = parser.max_times[agent_index],
                                            c=parser.c[agent_index],
                                            logs=parser.agent_logs[agent_index],
                                            name = "MCTS_c" + str(parser.c[agent_index]) if parser.names[agent_index] is None else parser.names[agent_index],
                                            rollouts=parser.rollouts[agent_index])
            
        elif agent_name == "eamcts":
            players[agent_index] = siea_mcts.SIEA_MCTS_Player(max_iterations = parser.iterations[agent_index], 
                                                max_time = parser.max_times[agent_index],
                                            logs = parser.agent_logs[agent_index],
                                            name = "EA_MCTS" if parser.names[agent_index] is None else parser.names[agent_index],
                                            rollouts=parser.rollouts[agent_index],
                                            es_lambda = es_lambda,
                                            es_fitness_iterations = es_fitness_iterations,
                                            es_generations = es_generations,
                                            use_semantics=False)
        elif agent_name == "sieamcts":
            players[agent_index] = siea_mcts.SIEA_MCTS_Player(max_iterations = parser.iterations[agent_index], 
                                                max_time = parser.max_times[agent_index],
                                            logs = parser.agent_logs[agent_index],
                                            name = "SIEA_MCTS" if parser.names[agent_index] is None else parser.names[agent_index],
                                            rollouts=parser.rollouts[agent_index],
                                            es_lambda = es_lambda,
                                            es_fitness_iterations = es_fitness_iterations,
                                            es_generations = es_generations,
                                            use_semantics=True)
        
        elif agent_name == "minimax":
            players[agent_index] = minimax.Minimax(max_depth = parser.minimax_max_depth[agent_index],
                                    name = "Minimax" if parser.names[agent_index] is None else parser.names[agent_index],
                                    star_L=parser.minimax_star_L[agent_index],
                                    star_U=parser.minimax_star_U[agent_index],
                                    probing_factor = parser.minimax_probing_factor[agent_index],
                                    heuristic_function = parser.heuristic_function[agent_index],
                                    move_ordering_function = parser.move_ordering_function[agent_index],
                                    logs=parser.agent_logs[agent_index])
            
        elif agent_name == "random":
            players[agent_index] = arand.RandomPlayer(name="Random" if parser.names[agent_index] is None else parser.names[agent_index], logs=parser.agent_logs[agent_index])
        else: 
            raise ValueError("Agent not recognised:", agent_name)

    #Create gameplayer
    gp1, gp2 = eu.play_match(agents=players,game_state = initial_game_states[0], games = parser.games, file_path = this_file_path, random_seed=parser.seed, logs = True)

    #gameplayer.save_data(file_path=file_path)
    
if __name__ == "__main__":
    datetime_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_file_path = os.path.join("Outputs","Carcassonne_2p", "Parameter_tuning")# datetime_string)  ##DONT FORGET TO CHANGE THIS
    es_lambda = 4
    es_fitness_iterations = 30
    es_generations = 20
    evolution_iterations = es_fitness_iterations*es_generations*es_lambda + es_fitness_iterations
    parsers = []
    games = 20
    rd_seed = 0
    iterations = 5000
    #move_time = 15

    include_pt = False
    #define parsers
    #file_name should be unique for each job
    #parser for parameter tuning
    if include_pt:
        parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mctsc0_5_random", agent1="mcts", agent2="random", logs1=True, logs2=True, file_path=default_file_path, c1=0.5))
        parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mctsc1_0_random", agent1="mcts", agent2="random", logs1=True, logs2=True, file_path=default_file_path, c1=1.0))
        parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mctsc1_41_random", agent1="mcts", agent2="random", logs1=True, logs2=True, file_path=default_file_path, c1=math.sqrt(2)))
        parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mctsc2_0_random", agent1="mcts", agent2="random", logs1=True, logs2=True, file_path=default_file_path, c1=2.0))
        parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mcts_3_0_random", agent1="mcts", agent2="random", logs1=True, logs2=True, file_path=default_file_path, c1=3.0))

    parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mcts_eamcts", agent1="mcts", agent2="eamcts", logs1=True, logs2=True, file_path=default_file_path, c1=0.5))
    parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mcts_sieamcts", agent1="mcts", agent2="sieamcts", logs1=True, logs2=True, file_path=default_file_path, c1=0.5))
    parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="eamcts_sieamcts", agent1="eamcts", agent2="sieamcts", logs1=True, logs2=True, file_path=default_file_path))
    parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="eamcts_random", agent1="eamcts", agent2="random", logs1=True, logs2=True, file_path=default_file_path))
    parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="sieamcts_random", agent1="sieamcts", agent2="random", logs1=True, logs2=True, file_path=default_file_path))
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mcts_eamcts", agent1="mcts", agent2="eamcts", logs1=True, logs2=True, file_path=default_file_path))
    #parsers.append(ExperimentParser(seed=rd_seed, games=games, iterations1=iterations, iterations2=iterations, file_folder="mcts_sieamcts", agent1="mcts", agent2="sieamcts", logs1=True, logs2=True, file_path=default_file_path))
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