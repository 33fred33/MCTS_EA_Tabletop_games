import cProfile
import os
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
import Agents.random as arand
import Agents.vanilla_mcts as mcts 
import Agents.siea_mcts as siea_mcts
import Games.function_optimisation as fo
import Utilities.logs_management as lm


#FOP experiment
logs_path = os.path.join("Outputs","FO_single_decision_5000")
random_seed = 1
runs = 30
iterations = 5000
es_lambda = 4
es_fitness_iterations = 30
es_generations = 20
c_list = [0.5, 1, math.sqrt(2),2,3]
c_names = ["0_5", "1", "1_4142", "2", "3"]
agents = []
function_indexes = []
function_indexes = [0,1,2,3,4]
#function_indexes += [5,6,7,8,9]

agents = [mcts.MCTS_Player(max_iterations=iterations, 
                           logs=True, 
                           c=c, 
                           name = "MCTS_c" + c_names[i]) for i,c in enumerate(c_list)]
agents = agents + [siea_mcts.SIEA_MCTS_Player(max_iterations=its, 
                                         es_lambda=es_lambda, 
                                         es_fitness_iterations=es_fitness_iterations,
                                        es_generations=es_generations,
                                        es_semantics_l=0.1,
                                        es_semantics_u = 0.5,
                                        name = "SIEA_MCTS_its" + str(its),
                                         logs=True) for its in [iterations, iterations+(es_fitness_iterations*es_generations*es_lambda)]]
agents = agents + [siea_mcts.SIEA_MCTS_Player(max_iterations=its, 
                                         es_lambda=es_lambda, 
                                         es_fitness_iterations=es_fitness_iterations,
                                        es_generations=es_generations,
                                        name = "EA_MCTS_its" + str(its),
                                        use_semantics=False,
                                         logs=True) for its in [iterations, iterations+(es_fitness_iterations*es_generations*es_lambda)]]
                                        
for function_index in function_indexes:
    game_state = fo.GameState(function_index=function_index, n_players=1)
    game_state.set_initial_state()
    for agent in agents:
        action = eu.mcts_decision_analysis(game_state = game_state,
                                             mcts_player = agent,
                                             logs_path = os.path.join(logs_path,"Function_" + str(function_index), agent.name),
                                             runs = runs,
                                             random_seed = random_seed)   
