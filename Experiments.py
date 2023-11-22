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
import Agents.siea_mcts2 as siea_mcts2
import Games.function_optimisation as fo
import Utilities.logs_management as lm

start_time = time.time()

#FOP experiment
logs_path = os.path.join("Outputs","FO_single_decision_missing_leaf_node")
random_seed = 1234
np.random.seed(random_seed)
rd.seed(random_seed)
runs = 100
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

#Calculations
evolution_iterations = es_fitness_iterations*es_generations*es_lambda + es_fitness_iterations

#Agents

"""
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
                                         logs=True) for its in [iterations, iterations-evolution_iterations]]
"""
agents = agents + [siea_mcts2.SIEA_MCTS_Player2(max_iterations=its, 
                                         es_lambda=es_lambda, 
                                         es_fitness_iterations=es_fitness_iterations,
                                        es_generations=es_generations,
                                        es_semantics_l=0.1,
                                        es_semantics_u = 0.5,
                                        name = "SIEA2_MCTS_its" + str(its),
                                        use_semantics=False,
                                         logs=True) for its in [iterations, iterations-evolution_iterations]]
"""
agents = agents + [siea_mcts.SIEA_MCTS_Player(max_iterations=its, 
                                        unpaired_evolution = True, ###########################bewaree
                                         es_lambda=es_lambda, 
                                         es_fitness_iterations=es_fitness_iterations,
                                        es_generations=es_generations,
                                        es_semantics_l=0.1,
                                        es_semantics_u = 0.5,
                                        name = "SIEA_MCTSu_its" + str(its),
                                         logs=True) for its in [iterations, iterations-evolution_iterations]]
"""
#Run                          
for function_index in function_indexes:
    print("In function " + str(function_index))
    game_state = fo.GameState(function_index=function_index, n_players=1)
    game_state.set_initial_state()
    for agent in agents:
        print("In agent " + agent.name)
        action = eu.mcts_decision_analysis(game_state = game_state,
                                             mcts_player = agent,
                                             logs_path = os.path.join(logs_path,"Function_" + str(function_index), agent.name),
                                             runs = runs,
                                             random_seed = random_seed)   
        print("Time:" + str(time.time() - start_time))

#Logs
for file_name in ["evolution_logs.csv", "results.csv", "logs_by_run.csv"]:
    file_path_list = lm.find_log_files(file_name, logs_path)
    lm.combine_logs(logs_path, file_name, file_path_list)

#Evolved formulas analysis

data = pd.read_csv(os.path.join(logs_path, "logs_by_run.csv"))
evolved_formula_data = pd.DataFrame()
for agent in data["Player"].unique():
    if "EA" in agent:
        for f_index in data["Function_index"].unique():
            tdata = data[(data["Player"]==agent) & (data["Function_index"]==f_index)]
            fa_data = eu.evolved_formula_analysis(tdata)
            fa_data["Player"] = [agent]
            fa_data["Function_index"] = [f_index]
            evolved_formula_data = pd.concat([evolved_formula_data, fa_data])
evolved_formula_data.to_csv(os.path.join(logs_path, "evolved_formula_analysis.csv"))
