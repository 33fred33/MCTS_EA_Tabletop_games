#select one puzzle
dataset_path = os.path.join("Datasets","pawnendgames_processed_lichess_db_puzzle.csv")
lichess_db = pd.read_csv(dataset_path)
puzzle_row = lichess_db.iloc[2]
#initialise chess
game_state = chess_64.GameState(breakthrough=True)
game_state.set_puzzle_lichess_db(puzzle_row)
print("Moves to look for:", puzzle_row["Moves"])
print(puzzle_row["Themes"])
print("Player to win:", "Black" if puzzle_row["Player_to_move"]==1 else "White")
#display board
board_image = chess.svg.board(game_state.board, size = 300)
display(board_image)
osla_agent = osla.OSLA_Wins()
random_agent = arand.RandomPlayer()
agents = [osla_agent, random_agent]
#evaluate with each agent
games = 100
rewards = {agent:[] for agent in agents}
for agent in agents:
    for game in range(games):
        game_log = eu.random_rollout(game_state, agent)
        rewards[agent].append(game_log["reward"])
    print(agent.name, np.mean(rewards[agent]))


#make moves from solution
solution = puzzle_row["Moves"].split(" ")[1:]
for move in solution:
    game_state.make_action(chess_64.Action(game_state.board.parse_uci(move), game_state.player_turn))
board_image = chess.svg.board(game_state.board, size = 300)
display(board_image)

#evaluate with each agent
games = 100
rewards = {agent:[] for agent in agents}
for agent in agents:
    for game in range(games):
        game_log = eu.random_rollout(game_state, agent)
        rewards[agent].append(game_log["reward"])
        #if game_log["winner"] is not None:
        #    board_image = chess.svg.board(game_log["final_state"].board, size = 300)
        #    display(board_image)

    print(agent.name, np.mean(rewards[agent]))
	
	
####next cell

#Chess experiment

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
import chess.svg
import matplotlib.pyplot as plt
import itertools
from IPython.display import display
import ast

#cProfile.run("wins =  random_games(10000, base_gs)")
#ut.run()

#Chess puzzle experiment

agent_rollouts = 1
max_iterations = 10000
iterations_logs_step = 500
random_seed = 0
runs = 1
#experiment named with date and time
experiment_path = os.path.join("Outputs", "Chess_puzzles_results_" + time.strftime("%Y%m%d-%H%M%S"))

#set random seed
rd.seed(random_seed)
np.random.seed(random_seed)

#experiment

#agent
#mcts_agent = mcts.MCTS_Player(max_iterations = max_iterations, rollouts=agent_rollouts)
mcts_agent = siea_mcts.SIEA_MCTS_Player(max_iterations = max_iterations, rollouts=agent_rollouts)

#dataset
#lichess_db = pd.read_csv("Datasets/lichess_1000_most_played.csv")
lichess_db = pd.read_csv("Datasets/final_lichess_db_puzzle.csv")

for row_number, (puzzle_idx, puzzle_row) in enumerate(lichess_db.iterrows()):
    #if row_number <= 3: #####################################################do first 2 puzzles (test)
        game_state = chess_64.GameState()
        try:
            game_state.set_puzzle_lichess_db(puzzle_row)
        except:
            continue

        results, parameters = eu.chess_puzzle_test(puzzle_row=puzzle_row, agent=mcts_agent, iterations_logs_step=iterations_logs_step, runs = runs, continue_moves=False)
        lm.dump_data(results, file_path= os.path.join(experiment_path, "Puzzle_"+ str(puzzle_idx)), file_name="results_every_"+ str(iterations_logs_step) +".csv")
        lm.dump_data(parameters, file_path= os.path.join(experiment_path, "Puzzle_"+ str(puzzle_idx)), file_name="experiment_data.csv")
        #board_image = chess.svg.board(game_state.board, size = 300)
        #display(board_image)
        #save board image svg
        #with open(os.path.join(experiment_path, "board_image.svg"), "w") as f:
        #    f.write(board_image)



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
    collective_results = pd.concat([collective_results, result_logs], axis=0)
    
    #End of loop routine
    puzzle_idx += 1

#Save collective results
collective_results.to_csv(os.path.join(experiment_path, "collective_results.csv"))

### next cell


#read csv results and plot

puzzle_number = rd.randint(0,242)
results_path = os.path.join("Outputs", "Chess_puzzles_results_20230922-172533", "Puzzle_" + str(puzzle_number))
file_name =  "results_every_500.csv"
file_df = pd.read_csv(os.path.join(results_path, file_name))
nonsolution_dict = {"Visits":[], "Rewards":[], "Iterations":[]}
for iterations in file_df["iterations_executed"].unique():
    nonsolutions_visits_df = file_df[file_df["iterations_executed"]==iterations]["nonsolution_visits_list"]
    nonsolutions_visits_list = [ast.literal_eval(nonsolution_visits) for nonsolution_visits in nonsolutions_visits_df]
    nonsolutions_visits_list = list(it.chain.from_iterable(nonsolutions_visits_list))
    nonsolution_dict["Visits"] += [v/iterations for v in nonsolutions_visits_list]
    #same for average rewards
    nonsolutions_rew_df = file_df[file_df["iterations_executed"]==iterations]["nonsolution_rewards_list"]
    nonsolutions_rew_list = [ast.literal_eval(nonsolution_rew) for nonsolution_rew in nonsolutions_rew_df]
    nonsolutions_rew_list = list(it.chain.from_iterable(nonsolutions_rew_list))
    nonsolution_dict["Rewards"] += nonsolutions_rew_list

    assert len(nonsolutions_visits_list) == len(nonsolutions_rew_list), "visits and rewards lists have different lengths"

    nonsolution_dict["Iterations"] += [iterations for i in range(len(nonsolutions_visits_list))]
nonsolution_df = pd.DataFrame(nonsolution_dict)
#nonsolution_df.to_csv(os.path.join(results_path, "nonsolutions.csv"))
#generate boxplot with plotly graphobjects


for criteria in ["Visits", "Rewards"]:
    fig = go.Figure()
    fig.add_trace(go.Box(y=nonsolution_df[criteria], name=criteria, x = nonsolution_df["Iterations"]))
    fig.update_layout(title="Nonsolutions " + criteria, xaxis_title="Iterations", yaxis_title=criteria)
    #Add solution values as scatter line
    if criteria == "Visits":
        y_visits = []
        y_nonsolution_most_visited_visits = []
        for i in range(len(file_df)):
            y_visits.append(file_df["solution_visits"][i]/file_df["iterations_executed"][i])
            y_nonsolution_most_visited_visits.append(file_df["nonsolution_most_visited_visits"][i]/file_df["iterations_executed"][i])
        fig.add_trace(go.Scatter(x=file_df["iterations_executed"], y=y_visits, mode="lines", name="Solution"))
        fig.add_trace(go.Scatter(x=file_df["iterations_executed"], y=y_nonsolution_most_visited_visits, mode="lines", name="Max Visits"))
    elif criteria == "Rewards":
        fig.add_trace(go.Scatter(x=file_df["iterations_executed"], y=file_df["solution_avg_reward"], mode="lines", name="Solution"))
        fig.add_trace(go.Scatter(x=file_df["iterations_executed"], y=file_df["nonsolution_highest_reward_avg_reward"], mode="lines", name="Max Reward"))
    #adjust fig size
    fig.update_layout(height=400, width=800)
    fig.show()

game_state = chess_64.GameState()
lichess_db = pd.read_csv("Datasets/final_lichess_db_puzzle.csv")
game_state.set_puzzle_lichess_db(lichess_db.iloc[puzzle_number])
board_image = chess.svg.board(game_state.board, size = 300)
display(board_image)
#print the tags
file_df = pd.read_csv(os.path.join(results_path, "experiment_data.csv"))
print(file_df["Themes"][0])
print(file_df["Moves"][0])

### next cell

#Experiment chess default policy

def default_policy_puzzles(puzzles, agents, games):
    for row_number, puzzle_row in enumerate(puzzles):
        game_state = chess_64.GameState()
        game_state.set_puzzle_lichess_db(puzzle_row)
        board_image = chess.svg.board(game_state.board, size = 300)
        display(board_image)
        print(puzzle_row["Themes"])
        print(puzzle_row["Moves"])
        
        final_df = pd.DataFrame()
        for agent in agents:
            start_time = time.time()
            fm_calls= []
            game_length = []
            rewards = []
            total_decisive = 0
            white_wins = 0
            corrects = 0
            for game_id in range(games):
                duplicate_state = game_state.duplicate()
                total_fm = 0
                turns = 0
                while not duplicate_state.is_terminal:
                    move = agent.choose_action(duplicate_state)
                    duplicate_state.make_action(move)
                    total_fm += agent.current_fm + 1
                    turns += 1
                if duplicate_state.winner is not None:
                    total_decisive += 1
                    if duplicate_state.winner == 0:
                        white_wins += 1
                    if duplicate_state.winner == game_state.player_turn:
                        corrects += 1
                fm_calls.append(total_fm)
                game_length.append(turns)
                rewards.append(duplicate_state.reward[game_state.player_turn])

            
            correct_in_decisive_ratio = corrects/total_decisive if total_decisive != 0 else None
            dict_1 = {"agent": [agent.name], "avg_fm_calls": [st.mean(fm_calls)], "std_fm_calls":[st.stdev(fm_calls)], "avg_game_length": [st.mean(game_length)], "std_game_length":[st.stdev(game_length)], "games": [games], "decisive_rate": [total_decisive/games], "white_wins":[white_wins/games], "avg_game_time": [(time.time() - start_time)/games], "correct_in_decisive_ratio": [correct_in_decisive_ratio],"correct_in_all_ratio": [corrects/games], "player_to_win": [game_state.player_turn], "rewards": [st.mean(rewards)], "std_rewards": [st.stdev(rewards)]}

            #checking the solution
            start_time = time.time()
            fm_calls= []
            game_length = []
            rewards = []
            total_decisive = 0
            white_wins = 0
            corrects = 0
            for game_id in range(games):
                duplicate_state = game_state.duplicate()
                for move in puzzle_row["Moves"].split(" ")[1:]:
                    duplicate_state.make_action(chess_64.Action(duplicate_state.board.parse_uci(move), duplicate_state.player_turn))
                total_fm = 0
                turns = 0
                while not duplicate_state.is_terminal:
                    move = agent.choose_action(duplicate_state)
                    duplicate_state.make_action(move)
                    total_fm += agent.current_fm + 1
                    turns += 1
                if duplicate_state.winner is not None:
                    total_decisive += 1
                    if duplicate_state.winner == 0:
                        white_wins += 1
                    if duplicate_state.winner == game_state.player_turn:
                        corrects += 1
                fm_calls.append(total_fm)
                game_length.append(turns)
                rewards.append(duplicate_state.reward[game_state.player_turn])
            
            correct_in_decisive_ratio = corrects/total_decisive if total_decisive != 0 else None
            dict_2 = {"from_solution_avg_fm_calls": [st.mean(fm_calls)], "from_solution_std_fm_calls":[st.stdev(fm_calls)], "from_solution_avg_game_length": [st.mean(game_length)], "from_solution_std_game_length":[st.stdev(game_length)], "from_solution_decisive_rate": [total_decisive/games], "from_solution_white_wins":[white_wins/games], "from_solution_avg_game_time": [(time.time() - start_time)/games], "from_solution_correct_in_decisive_ratio": [correct_in_decisive_ratio],"from_solution_correct_in_all_ratio": [corrects/games], "from_solution_rewards": [st.mean(rewards)], "from_solution_std_rewards": [st.stdev(rewards)]}
            #combine dicts
            dict_1.update(dict_2)
            df = pd.DataFrame(dict_1)

            df = pd.concat([df, agent.agent_data()], axis = 1)
            df = pd.concat([df, game_state.logs_data()], axis = 1)
            row_frame = puzzle_row.to_frame().T
            row_frame.index = [0]
            df = pd.concat([df, row_frame], axis = 1)
            final_df = pd.concat([final_df, df])
        print("Finished puzzle " + str(puzzle_row["PuzzleId"]) + "_rownumber_" + str(row_number))
        #final_df.to_csv(os.path.join("Outputs","Comparison_default_policies","Chess_" + str(puzzle_row["PuzzleId"]) + "_games" + str(games) + ".csv"), index = False)
        lm.dump_data(data = final_df, file_path = os.path.join("Outputs","Comparison_default_policies_endgames"), file_name = "Chess_Puzzle_" + str(puzzle_row["PuzzleId"]) + "_games" + str(games) + "_2.csv")

lichess_db = pd.read_csv("Datasets/pawnendgames_processed_lichess_db_puzzle.csv")
puzzles = []
puzzles_idx = list(range(10))
if len(puzzles_idx) > 0:
    for pidx in puzzles_idx:
        puzzles.append(lichess_db.iloc[pidx])
else:
    for i in range(len(lichess_db)):
        puzzles.append(lichess_db.iloc[i])
agents = [osla.OSLA_Wins(max_probing_size=1), osla.OSLA_Wins(max_probing_size=2), osla.OSLA_Wins(), arand.RandomPlayer()]
games = 100

default_policy_puzzles(puzzles, agents, games)
#cProfile.run("default_policy_puzzles(puzzles, agents, games)")
#game_state = chess_64.GameState()
#game_state.set_puzzle_lichess_db(puzzles[0])
#print(puzzles[0]["FEN"])
#print(puzzles[0]["PuzzleId"])
#game_state.board

### next cell

#Puzzle search
by_criteria=False
#tags_to_include = ["crushing", "pawnEndgame"]
tags_to_include = ["endgame", "equality"]
tags_to_remove = ["oneMove"]
#tags_to_remove = ["middlegame","opening","mateIn5","mateIn1","oneMove","veryLong"]

#theme = "pawnEndgame"
#theme = "pin"
next_puzzle = 0
#random_one = True
puzzle_idx = 0
#puzzle_id = "000Vc"
puzzle_id = None

#lichess_db = pd.read_csv("Datasets/lichess_1000_most_played.csv")
#lichess_db = pd.read_csv("Datasets/lichess_db_puzzle_subsample.csv")
#lichess_db = pd.read_csv("Datasets/processed_lichess_db_puzzle.csv")
lichess_db = pd.read_csv("Datasets/chosen_processed_lichess_db_puzzle.csv")
#lichess_db = pd.read_csv("Datasets/pawnendgames_processed_lichess_db_puzzle.csv")

puzzles_found = -1
if by_criteria:
    for puzzle_idx, puzzle_row in lichess_db.iterrows():
        row_tags = puzzle_row["Themes"].split(" ")
        if len(set(row_tags).intersection(set(tags_to_include))) == len(tags_to_include):
            if len(set(row_tags).intersection(set(tags_to_remove))) == 0:
                #print(row_tags)
                puzzles_found += 1
                if puzzles_found == next_puzzle:
                    print("Puzzle found at index:", puzzle_idx)
                    break
else:
    if puzzle_id is not None:
        puzzle_row = lichess_db.loc[lichess_db["PuzzleId"] == puzzle_id]
        puzzle_idx = puzzle_row.index[0]
        print("Puzzle found at index:", puzzle_idx)
    else:
        puzzle_row = lichess_db.iloc[puzzle_idx]
        
game_state = chess_64.GameState()
#puzzle_row = lichess_db.iloc[puzzle_idx]
game_state.set_puzzle_lichess_db(puzzle_row)
#print("Move to look for:", puzzle_row["Moves"].split(" ")[1])
#print("Moves to look for:", puzzle_row["Moves"])
print(str(puzzle_row))
print(puzzle_row["Moves"])
print(puzzle_row["Themes"])
board_image = chess.svg.board(game_state.board, size = 300)
display(board_image)
#save board image svg
#with open(os.path.join("Outputs", "board_image.svg"), "w") as f:
#    f.write(board_image)


### next cell

#Create lichess db subsets
dataset_file = "processed_lichess_db_puzzle.csv"
dataset_path = "Datasets"
lichess_db = pd.read_csv(os.path.join(dataset_path, dataset_file))
#max_puzzles = 100
#min_per_tag = 10

#filtering by tags
tags_to_avoid = ["mateIn1", "oneMove", "opening", "mateIn2", "mateIn3", "mateIn4", "mateIn5", "mate"]
tags_to_include = ["pawnEndgame"]#["mateIn5","mateIn4","mateIn3","mateIn2"]#,
#tags_to_include_after = ["endgame"]
tags_to_include_after = ["crushing", "equality", "sacrifice"]

add_list = [True for _ in range(len(lichess_db))]
for row_idx, row in lichess_db.iterrows():
    row_tags = row["Themes"].split(" ")

    #tags to include
    if len(set(row_tags).intersection(set(tags_to_include))) < 1:
        add_list[row_idx] = False
        continue

    if len(set(row_tags).intersection(set(tags_to_include_after))) < 1:
        add_list[row_idx] = False
        continue

    #tags to avoid
    for tag in tags_to_avoid:
        if tag in row_tags:
            add_list[row_idx] = False
            break
    if not add_list[row_idx]:
        continue
lichess_db = lichess_db[add_list]
print(len(lichess_db), "puzzles after filtering by tags")

#Filtering by other
lichess_db = lichess_db[lichess_db["Pieces"] < 10]
print(len(lichess_db), "puzzles after filtering by pieces")
lichess_db = lichess_db[lichess_db["Available_actions"] < 10]
print(len(lichess_db), "puzzles after filtering by available_actions")
lichess_db = lichess_db[lichess_db["NbPlays"] > 1000]
print(len(lichess_db), "puzzles after filtering by NbPlays")
lichess_db = lichess_db[lichess_db["Required_moves"] <= 3]
print(len(lichess_db), "puzzles after filtering by Required_moves")

#save subset to csv
new_dataset_name = "pawnendgames_" + dataset_file
lm.dump_data(data = lichess_db, file_path = dataset_path, file_name = new_dataset_name)

#analyse the dataset by tag
tag_analysis = eu.analyse_puzzles_by_tag(lichess_db)
lm.dump_data(data = tag_analysis, file_path = dataset_path, file_name = "tags_of_" + new_dataset_name)

### next cell


#Create lichess db subsets detailed
dataset_file = "processed_lichess_db_puzzle.csv"
#dataset_file = "lichess_db_puzzle_subsample.csv"
dataset_path = "Datasets"
lichess_db = pd.read_csv(os.path.join(dataset_path, dataset_file))

#Filtering by other
lichess_db = lichess_db[lichess_db["Pieces"] < 10]
print(len(lichess_db), "puzzles after filtering by pieces")
lichess_db = lichess_db[lichess_db["Available_actions"] < 10]
print(len(lichess_db), "puzzles after filtering by available_actions")
lichess_db = lichess_db[lichess_db["NbPlays"] > 100]

#filtering by tags
tags_to_include = ["pawnEndgame"]
tags_to_avoid = ["mateIn1", "oneMove"]
#tags_variety = [["short", "long", "veryLong"],["endgame", "middlegame", "opening"],["mate","crushing","equality"]]
tags_variety = [["short", "long"],["mate","crushing","equality"]]
rating_breakpoints = [1500, 2000, 2500]
sorting_criteria = ["Available_actions", "Pieces","Rating"]
ascending = [True, True, False]

#initial_filters
previous_len = len(lichess_db)
#lichess_db = lichess_db[lichess_db["NbPlays"] > 1000]
#print(len(lichess_db), "puzzles after filtering by NbPlays (",len(lichess_db)/previous_len,")")
for tag_to_avoid in tags_to_avoid:
    lichess_db = lichess_db[~lichess_db["Themes"].str.contains(tag_to_avoid)]
print(len(lichess_db), "puzzles after filtering by tags to avoid")
#iterate by tag variety
tag_combinations = list(itertools.product(*tags_variety))
all_combinations = [(*tag_combination, rating_breakpoint) for rating_breakpoint in rating_breakpoints for tag_combination in tag_combinations]
#tag_combination_add_list = {tag_combination: [] for tag_combination in tag_combinations}
all_combinations_add_list = {combination: [] for combination in all_combinations}
#print("Tag combinations:", len(tag_combinations))
print("All combinations:", len(all_combinations))
#print("All combinations legnth:", len(list(all_combinations_add_list.values())[0]))
#print("All combinations keys:", all_combinations_add_list.keys())

for idx, (row_idx, row) in enumerate(lichess_db.iterrows()):
    row_tags = row["Themes"].split(" ")
    if len(set(row_tags).intersection(set(tags_to_include))) < 1:
        continue
    for tag_combination in tag_combinations:
        for rating_breakpoint in rating_breakpoints:
            if row["Rating"] < rating_breakpoint and row["Rating"] >= rating_breakpoint - 500:
                if len(set(row_tags).intersection(set(tag_combination))) == len(tag_combination):
                    if "veryLong" in tag_combination:
                        if row["Required_moves"] == 4:
                            all_combinations_add_list[(*tag_combination, rating_breakpoint)].append(row_idx)
                    else:
                        all_combinations_add_list[(*tag_combination, rating_breakpoint)].append(row_idx)
                    

#get dataset subset
final_df = pd.DataFrame()
for key, value in all_combinations_add_list.items():
    lichess_subset = lichess_db.loc[value]
    print(key, len(lichess_subset))
    #sorting
    for sorting_criterion, asc in zip(sorting_criteria, ascending):
        lichess_db = lichess_db.sort_values(by=[sorting_criterion], ascending=asc)
    #selecting
    lichess_subset = lichess_subset.iloc[0:3]
    final_df = pd.concat([final_df, lichess_subset])

extended_file_name = "ffinal_" + dataset_file
lm.dump_data(data = final_df, file_path = dataset_path, file_name = extended_file_name)
tag_analysis = eu.analyse_puzzles_by_tag(final_df)
lm.dump_data(data = tag_analysis, file_path = dataset_path, file_name = "tags_of_" + extended_file_name)


### next cell


#Extend dataset with devalue analysis

lichess_db = pd.read_csv("Datasets/pawnendgames_processed_lichess_db_puzzle.csv")
#get only first 3 rows of db
#lichess_db = lichess_db.iloc[0:3]
game_state = chess_64.GameState()
devalue_rollouts = 1000
Initial_devalue = [None for _ in range(len(lichess_db))]
Average_fm_calls = [None for _ in range(len(lichess_db))]
Average_decisive_results = [None for _ in range(len(lichess_db))]
Solution_devalue = [None for _ in range(len(lichess_db))]
Solution_avg_fm_calls = [None for _ in range(len(lichess_db))]
Solution_avg_decisive_results = [None for _ in range(len(lichess_db))]
Solution_final_devalue = [None for _ in range(len(lichess_db))]
Solution_final_avg_fm_calls = [None for _ in range(len(lichess_db))]
Solution_final_avg_decisive_results = [None for _ in range(len(lichess_db))]
Average_branching_factor = [None for _ in range(len(lichess_db))]
Stdev_branching_factor = [None for _ in range(len(lichess_db))]
Maximum_iterations_to_expand_solution = [None for _ in range(len(lichess_db))]
Minimum_needed_nodes = [None for _ in range(len(lichess_db))] #nodes needed to reach solution
Minimum_needed_nodes_and_final_available = [None for _ in range(len(lichess_db))] #nodes needed to reach solution + final available actions
Opponent_options_ignored = [None for _ in range(len(lichess_db))] #actions that wont be considered by solution
Opponent_total_options_in_solution = [None for _ in range(len(lichess_db))] #actions that opponent had to consider in solution
osla_agent = osla.OSLA_Wins()

for row_number, (puzzle_idx, puzzle_row) in enumerate(lichess_db.iterrows()):
    if row_number % 10 == 0:
        print("Puzzle:", row_number)
    game_state.set_puzzle_lichess_db(puzzle_row)
    puzzle_moves = puzzle_row["Moves"].split(" ")[1:]
    branching_factor_scope = puzzle_row["Required_moves"] * 2 - 1
    #duplicate_state = game_state.duplicate()

    result_holder = []
    total_fm_calls = 0
    decisive_results_count = 0
    branching_factors = []
    for _ in range(devalue_rollouts):
        duplicate_state = game_state.duplicate()
        depth = 0
        while duplicate_state.is_terminal == False:
            action = rd.choice(duplicate_state.available_actions)
            duplicate_state.make_action(action)
            total_fm_calls += 1
            depth += 1
            if depth <= branching_factor_scope:
                branching_factors.append(len(duplicate_state.available_actions))
        result_holder.append(duplicate_state.reward[game_state.player_turn])
        if duplicate_state.winner is not None:
            decisive_results_count += 1

    Initial_devalue[row_number] = st.mean(result_holder)
    Average_fm_calls[row_number] = total_fm_calls/devalue_rollouts
    Average_decisive_results[row_number] = decisive_results_count/devalue_rollouts
    Average_branching_factor[row_number] = st.mean(branching_factors)
    Stdev_branching_factor[row_number] = st.stdev(branching_factors)
    Maximum_iterations_to_expand_solution[row_number] = st.mean(branching_factors)**branching_factor_scope

    next_move = puzzle_moves.pop(0)
    game_state.make_action(chess_64.Action(game_state.board.parse_uci(next_move), player_turn=game_state.player_turn))

    result_holder = []
    total_fm_calls = 0
    decisive_results_count = 0
    branching_factors = []
    for _ in range(devalue_rollouts):
        duplicate_state = game_state.duplicate()
        depth = 0
        while duplicate_state.is_terminal == False:
            action = rd.choice(duplicate_state.available_actions)
            duplicate_state.make_action(action)
            total_fm_calls += 1
            depth += 1
            if depth <= branching_factor_scope:
                branching_factors.append(len(duplicate_state.available_actions))
        result_holder.append(duplicate_state.reward[game_state.player_turn])
        if duplicate_state.winner is not None:
            decisive_results_count += 1
    
    Solution_devalue[row_number] = st.mean(result_holder)
    Solution_avg_fm_calls[row_number] = total_fm_calls/devalue_rollouts
    Solution_avg_decisive_results[row_number] = decisive_results_count/devalue_rollouts

    Minimum_needed_nodes[row_number] = len(game_state.available_actions)
    Minimum_needed_nodes_and_final_available[row_number] = len(game_state.available_actions)
    Opponent_options_ignored[row_number] = len(game_state.available_actions) - 1
    Opponent_total_options_in_solution[row_number] = len(game_state.available_actions)
    for move in puzzle_moves:
        game_state.make_action(chess_64.Action(game_state.board.parse_uci(move), player_turn=game_state.player_turn))
        if game_state.player_turn == puzzle_row["Player_to_move"]:
            Opponent_options_ignored[row_number] += len(game_state.available_actions) - 1
            Opponent_total_options_in_solution[row_number] += len(game_state.available_actions)
        #if not the last move in the loop, append
        Minimum_needed_nodes_and_final_available[row_number] += len(game_state.available_actions)
        if move != puzzle_moves[-1]:
            Minimum_needed_nodes[row_number] += len(game_state.available_actions)

    result_holder = []
    total_fm_calls = 0
    decisive_results_count = 0
    branching_factors = []
    for _ in range(devalue_rollouts):
        duplicate_state = game_state.duplicate()
        depth = 0
        while duplicate_state.is_terminal == False:
            action = rd.choice(duplicate_state.available_actions)
            duplicate_state.make_action(action)
            total_fm_calls += 1
            depth += 1
            if depth <= branching_factor_scope:
                branching_factors.append(len(duplicate_state.available_actions))
        result_holder.append(duplicate_state.reward[game_state.player_turn])
        if duplicate_state.winner is not None:
            decisive_results_count += 1

    Solution_final_devalue[row_number] = st.mean(result_holder)
    Solution_final_avg_fm_calls[row_number] = total_fm_calls/devalue_rollouts
    Solution_final_avg_decisive_results[row_number] = decisive_results_count/devalue_rollouts

    if row_number == 10:
        break

lichess_db["Initial_devalue"] = Initial_devalue
lichess_db["Average_fm_calls"] = Average_fm_calls
lichess_db["Average_decisive_results"] = Average_decisive_results
lichess_db["Solution_devalue"] = Solution_devalue
lichess_db["Solution_final_devalue"] = Solution_final_devalue
lichess_db["Average_branching_factor"] = Average_branching_factor
lichess_db["Stdev_branching_factor"] = Stdev_branching_factor
lichess_db["Maximum_iterations_to_expand_solution"] = Maximum_iterations_to_expand_solution
lichess_db["Solution_avg_fm_calls"] = Solution_avg_fm_calls
lichess_db["Solution_avg_decisive_results"] = Solution_avg_decisive_results
lichess_db["Solution_final_avg_fm_calls"] = Solution_final_avg_fm_calls
lichess_db["Solution_final_avg_decisive_results"] = Solution_final_avg_decisive_results
lichess_db["Minimum_needed_nodes"] = Minimum_needed_nodes
lichess_db["Minimum_needed_nodes_and_final_available"] = Minimum_needed_nodes_and_final_available
lichess_db["Opponent_options_ignored"] = Opponent_options_ignored
lichess_db["Opponent_total_options_in_solution"] = Opponent_total_options_in_solution

#one hot enconding of relevant tags
relevant_tags = ["short", "long", "veryLong", "endgame", "middlegame", "opening", "mate", "crushing", "equality"]
for tag in relevant_tags:
    lichess_db[tag] = [0 for _ in range(len(lichess_db))]
for row_idx, row in lichess_db.iterrows():
    row_tags = row["Themes"].split(" ")
    for tag in relevant_tags:
        if tag in row_tags:
            lichess_db[tag][row_idx] = 1

#one hot encoding of rating ranges
relevant_rating_ranges = [(1000, 1500), (1500, 2000), (2000, 2500)]
for relevant_rating_range in relevant_rating_ranges:
    lichess_db["Rating_" + str(relevant_rating_range[0]) + "_" + str(relevant_rating_range[1])] = [0 for _ in range(len(lichess_db))]
for row_idx, row in lichess_db.iterrows():
    for relevant_rating_range in relevant_rating_ranges:
        if row["Rating"] >= relevant_rating_range[0] and row["Rating"] < relevant_rating_range[1]:
            lichess_db["Rating_" + str(relevant_rating_range[0]) + "_" + str(relevant_rating_range[1])][row_idx] = 1

#compare devalues
lichess_db["Final_solution_devalue_better_than_beginning"] = [False for _ in range(len(lichess_db))]
lichess_db["Final_solution_devalue_improvement"] = [None for _ in range(len(lichess_db))]
lichess_db["Final_solution_devalue_better_than_after_first_move"] = [False for _ in range(len(lichess_db))]
for row_idx, row in lichess_db.iterrows():
    lichess_db["Final_solution_devalue_improvement"][row_idx] = row["Solution_final_devalue"] - row["Initial_devalue"]
    if row["Solution_final_devalue"]>row["Initial_devalue"]:
        lichess_db["Final_solution_devalue_better_than_beginning"][row_idx] = True
    if row["Solution_final_devalue"]>row["Solution_devalue"]:
        lichess_db["Final_solution_devalue_better_than_after_first_move"][row_idx] = True

lichess_db.to_csv("Datasets/final_pawnendgames_lichess_db_puzzle.csv")


### next cell

#Extend and analyse lichess db

dataset_file = "lichess_db_puzzle.csv"
dataset_path = "Datasets"
#read db
lichess_db = pd.read_csv(os.path.join(dataset_path, dataset_file))
#extend lichess db
extended_db = eu.extend_lichess_db(dataset_path, dataset_file)
extended_file_name = "processed_" + dataset_file
lm.dump_data(data = extended_db, file_path = dataset_path, file_name = extended_file_name)
#analyse the dataset by tag
tag_analysis = eu.analyse_puzzles_by_tag(extended_db)
lm.dump_data(data = tag_analysis, file_path = dataset_path, file_name = "tags_of_" + extended_file_name)

### next cell

#Collect chess puzzles results - Already in experiment

running = True
puzzle_idx = 0
iterations_logs_step = 50
run = 0
experiment_path = os.path.join("Outputs", "Chess_puzzles")
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

### next cell

#Analyse puzzle results by tag

#Read collective results
experiment_path = os.path.join("Outputs", "Chess_puzzles")
experiment_path = os.path.join("Outputs", "Chess_puzzles_results_siea")
collective_results_df = pd.read_csv(os.path.join(experiment_path, "collective_results.csv"))

tag_dict = {}
all_tags = []
for row in collective_results_df.iterrows():
    tags = row[1]['Themes'].split(' ')
    for tag in tags:
        if tag not in all_tags:
            all_tags.append(tag)

for tag in all_tags:
    subset_df = collective_results_df[collective_results_df['Themes'].str.contains(tag)]
    tag_count = len(subset_df)
    correct_subset = subset_df[subset_df['solved_ratio'] == 1]
    tag_correct_count = len(correct_subset)
    tag_dict[tag] = [tag_count/len(collective_results_df),tag_count, tag_correct_count / tag_count]
    #get average_rating
    tag_dict[tag].append(subset_df['Rating'].mean())

tag_df = pd.DataFrame.from_dict(tag_dict, orient='index', columns=["appearance_ratio","count",'correct_ratio',"average_rating"])
tag_df.sort_values('correct_ratio', ascending=False, inplace=True)
tag_df.to_csv(os.path.join(experiment_path, "tags_of_collective_results.csv"))
print(tag_df)


### next cell


