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
#import Games.carcassonne_older as csn
import Agents.random as arand
import Agents.vanilla_mcts as mcts 
import Agents.siea_mcts as siea_mcts
import Agents.mcts_rave as rave_mcts
import Games.function_optimisation as fo
import Utilities.logs_management as lm
import Games.chess_64 as chess_64
import chess
import matplotlib.pyplot as plt

#cProfile.run("wins =  random_games(10000, base_gs)")
#ut.run()

#Chess puzzle experiment

agent_rollouts = 1
max_iterations = 5000
iterations_logs_step = 50
random_seed = 1234
experiment_path = os.path.join("Outputs", "Chess_puzzles_results_siea")

#set random seed
rd.seed(random_seed)
np.random.seed(random_seed)

#experiment

#agent
#mcts_agent = mcts.MCTS_Player(max_iterations = max_iterations, rollouts=agent_rollouts)
mcts_agent = siea_mcts.SIEA_MCTS_Player(max_iterations = max_iterations, rollouts=agent_rollouts)

#dataset
#lichess_db = pd.read_csv("Datasets/lichess_1000_most_played.csv")
lichess_db = pd.read_csv("Datasets/lichess_db_puzzle_subsample.csv")

for puzzle_idx, puzzle_row in lichess_db.iterrows():
    if puzzle_idx >= 93:
        game_state = chess_64.GameState()
        try:
            game_state.set_puzzle_lichess_db(puzzle_row)
        except:
            continue

        results, parameters = eu.chess_puzzle_test(puzzle_row=puzzle_row, agent=mcts_agent, iterations_logs_step=iterations_logs_step)
        lm.dump_data(results, file_path= os.path.join(experiment_path, "Puzzle_"+ str(puzzle_idx)), file_name="results_every_"+ str(iterations_logs_step) +".csv")
        lm.dump_data(parameters, file_path= os.path.join(experiment_path, "Puzzle_"+ str(puzzle_idx)), file_name="experiment_data.csv")



#Collect chess puzzles results

running = True
puzzle_idx = 0
run = 0
collective_results = pd.DataFrame()
while running:
    try:
        move_logs = pd.read_csv(os.path.join(experiment_path, "Puzzle_"+str(puzzle_idx), "results_every_"+ str(iterations_logs_step)+".csv"))
        result_logs = pd.read_csv(os.path.join(experiment_path, "Puzzle_"+str(puzzle_idx), "experiment_data.csv"))

    except:
        running = False
        break    

    run_logs = move_logs[move_logs["exp_run"]==run]
    final_row = run_logs.iloc[-1]
    iteration_as_last_choice = []
    first_action_solution = run_logs["expected_move"][0]
    final_last_unchanged_iteration = None
    if final_row["current_chosen_move"] == first_action_solution:
        correct_run_logs = run_logs[run_logs["current_chosen_move"]==first_action_solution]
        last_unchanged_iteration = correct_run_logs["iterations_executed"].max()
        #print("correct_run_logs_len",str(len(correct_run_logs)), "run_logs_len", str(len(run_logs)))
        for i in reversed(range(len(correct_run_logs))[1:]):
            if abs(correct_run_logs.iloc[i]["iterations_executed"] - correct_run_logs.iloc[i-1]["iterations_executed"]) == iterations_logs_step:
                last_unchanged_iteration = correct_run_logs.iloc[i-1]["iterations_executed"]
            else: break
        iteration_as_last_choice.append(last_unchanged_iteration)
        final_last_unchanged_iteration = st.mean(iteration_as_last_choice)
    
    #Add data to results log
    result_logs["final_last_unchanged_iteration"] = [final_last_unchanged_iteration]
    result_logs["puzzle_idx"] = [puzzle_idx]

    #Insert final row
    final_row_df = pd.DataFrame([final_row.tolist()], columns=final_row.index)

    #find repeated columns
    repeated_columns = []
    for column in final_row_df.columns:
        if column in result_logs.columns:
            repeated_columns.append(column)
    #drop repeated columns
    final_row_df = final_row_df.drop(repeated_columns, axis=1)
    
    result_logs = pd.concat([result_logs, final_row_df], axis=1)

    #Append data to collective results
    collective_results = pd.concat([collective_results, result_logs], axis=0, ignore_index=True)
    
    #End of loop routine
    puzzle_idx += 1

#Save collective results
collective_results.to_csv(os.path.join(experiment_path, "collective_results.csv"))