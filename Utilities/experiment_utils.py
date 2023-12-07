import cProfile
import random as rd
import time
from collections import defaultdict
import numpy as np
from itertools import cycle
import pandas as pd
import statistics as st

import Games.mnk as mnk
import Games.chess_64 as chess_64
import chess
import Agents.random as arand
import Agents.vanilla_mcts as mcts 
import os
import Utilities.logs_management as lm
import statistics as stats

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import colorsys
from IPython.display import HTML

class GamePlayer():
    def __init__(self, game_state, players) -> None:
        self.game_state = game_state
        self.players = players
        self.games_count = 0
        self.logs_by_game = pd.DataFrame()
        self.logs_by_action = pd.DataFrame()
        self.logs_by_iteration = pd.DataFrame()
        self.win_count = {i:0 for i in range(len(players))}
        self.win_count["Draw"] = 0
        
    def play_game(self, random_seed = None, logs = True, verbose = True):
        "Plays a game. If logs is true, adds data to the class logs"

        gs = self.game_state.duplicate()

        #Set random seed
        if random_seed is not None: 
            rd.seed(random_seed)
            np.random.seed(random_seed)
        else: 
            random_seed = rd.randint(0, 2**32)
            rd.seed(random_seed)
            np.random.seed(random_seed)

        #Set logs
        action_logs = pd.DataFrame()
        game_logs = pd.DataFrame()
        iteration_logs = pd.DataFrame()
        if logs:
            for p in self.players:
                p.logs = True

        #Play game
        game_start_time = time.time()
        safe_count = 0
        while not gs.is_terminal:
            start_time = time.time()
            action = self.players[gs.player_turn].choose_action(gs)
            selection_time = time.time() - start_time

            #Update logs
            if logs:
                action_log = self.players[gs.player_turn].choose_action_logs #Assumes this log is single row
                action_log_addon = {}
                action_log_addon["game_index"] = [self.games_count]
                action_log_addon["pg_selection_time"] = [selection_time]
                action_log_addon["pg_returned_action"] = [str(action)]
                action_log_addon["pg_player"] = [str(self.players[gs.player_turn])]
                action_log_addon["pg_player_index"] = [gs.player_turn]
                action_log_addon["pg_player_name"] = [self.players[gs.player_turn].name]
                action_log_addon["pg_game_turn"] = [gs.turn]
                action_log_addon_df = pd.DataFrame(action_log_addon)
                action_log = pd.concat([action_log, action_log_addon_df], axis=1)
                action_logs = pd.concat([action_logs, action_log], ignore_index=True)
                game_logs = pd.concat([game_logs, gs.logs_data()], ignore_index=True)

                #check if player has the attribute logs_by_iterations
                if hasattr(self.players[gs.player_turn], "logs_by_iterations"):
                    iteration_log = self.players[gs.player_turn].logs_by_iterations
                    iteration_log_addon = {k:[v for _ in range(len(iteration_log))] for k,v in action_log_addon.items()}
                    iteration_log_addon_df = pd.DataFrame(iteration_log_addon)
                    iteration_log = pd.concat([iteration_log, iteration_log_addon_df], axis=1)
                    iteration_logs = pd.concat([iteration_logs, iteration_log], ignore_index=True)



            #safety check      
            safe_count += 1
            if safe_count > 1000: 
                print("Safe count exceeded")
                print(gs.logs_data())
                print("Last action:" + str(action))
                break
            assert safe_count < 1000, "Safe count exceeded"

            #Make action
            gs.make_action(action)

            if verbose:
                print("Player " + str(gs.player_turn) + " chose action " + str(action) + " in " + str(selection_time) + " seconds")

            #if gs.turn > 2: break ####TEMPORARY
        
        if logs:
            #Final logs by action
            final_logs_by_action = pd.concat([action_logs, game_logs], axis=1)


            #Final logs by game
            final_logs_by_game_dict = {}
            for i, player in enumerate(self.players):
                final_logs_by_game_dict["Player_" + str(i)] = str(player)
            final_logs_by_game_dict["game_random_seed"] = random_seed
            final_logs_by_game_dict["game_index"] = self.games_count
            final_logs_by_game_dict["game_time"] = str(time.time() - game_start_time)
            final_logs_by_game = pd.DataFrame(final_logs_by_game_dict, index=[0])
            final_logs_by_game = pd.concat([final_logs_by_game, gs.logs_data()], axis=1)
            
            #Update class logs
            self._update_logs_by_game(final_logs_by_game)
            self._update_logs_by_action(final_logs_by_action)
            if len(iteration_logs) > 0:
                self._update_logs_by_iteration(iteration_logs)
            self._update_win_count(gs.winner)

        self.games_count += 1
        return gs

    def _update_win_count(self, winner):
        if winner is None: self.win_count["Draw"] += 1
        else:
            self.win_count[winner] += 1

    def _update_logs_by_game(self, logs):
        self.logs_by_game = pd.concat([self.logs_by_game, logs], ignore_index=True)

    def _update_logs_by_action(self, logs):
        self.logs_by_action = pd.concat([self.logs_by_action, logs], ignore_index=True)

    def _update_logs_by_iteration(self, logs):
        self.logs_by_iteration = pd.concat([self.logs_by_iteration, logs], ignore_index=True)

    def play_games(self, n_games, random_seed = None, logs = True, logs_dispatch_after = 1, logs_path = None, verbose = False) -> None:
        "Plays n_games games"
        if random_seed is not None: 
            rd.seed(random_seed)
            np.random.seed(random_seed)
        else: 
            random_seed = rd.randint(0, 2**32)
            rd.seed(random_seed)
            np.random.seed(random_seed)
        seeds = rd.sample(range(0, 2**32), n_games)
        for i in range(n_games):
            if verbose: print("Playing game " + str(i+1) + " of " + str(n_games) + ", agents: " + str([a.name for a in self.players]))
            self.play_game(random_seed = seeds[i], logs = logs, verbose = verbose)
            if logs:
                if i % logs_dispatch_after == 0:
                    self.save_data(logs_path)
        if logs: self.save_data(logs_path)

    def save_data(self, file_path):
        "Saves logs to file_path"
        lm.dump_data(self.logs_by_game, file_path= file_path, file_name="logs_by_game.csv")
        lm.dump_data(self.game_state.game_definition_data(), file_path= file_path, file_name="game_definition_data.csv")
        lm.dump_data(self.logs_by_action, file_path= file_path, file_name="logs_by_action.csv")
        lm.dump_data(self.results(), file_path= file_path, file_name="results.csv")
        for i,p in enumerate(self.players):
            lm.dump_data(p.agent_data(), file_path= file_path, file_name="p" + str(i) + "_data.csv")
        if len(self.logs_by_iteration) > 0:
            lm.dump_data(self.logs_by_iteration, file_path= file_path, file_name="logs_by_iteration.csv")

    def results(self):
        "Returns a dictionary with the results of the games played"
        results = {}
        results["games_count"] = self.games_count
        results["game_name"] = self.game_state.name
        for i in range(len(self.players)):
            results["Player_" + str(i)] = str(self.players[i])
            results["win rate_" + str(i)] = self.win_count[i] / self.games_count
        results["draw rate"] = self.win_count["Draw"] / self.games_count
        results = pd.DataFrame(results, index=[0])
        return results

    def __str__(self):
        return_string = "GamePlayer_Game_" + self.game_state.name
        for i, p in enumerate(self.players):
            return_string += "_P" + str(i) + "_" + p.name
        return return_string


def play_match(agents, game_state, games, file_path = None, random_seed = None, logs = True, logs_dispatch_after = 1):
    """Plays a match between two agents
    agents: list of agents
    game_state: game state
    games: number of games to play. Total played games will be 2*games
    file_path: path to save logs
    random_seed: random seed for the match
    logs: if True, logs will be saved
    logs_dispatch_after: number of games to play before saving logs (for security). Only used if logs is True. Logs will be stored at the end of the match anyway
    """
    if random_seed is None: random_seed = rd.randint(0, 2**32)
    gp1 = GamePlayer(game_state.duplicate(), agents)
    subfolder = "1_"+ gp1.players[0].name + "_vs_" + gp1.players[1].name
    gp1.play_games(games, random_seed = random_seed, logs = logs, logs_path=os.path.join(file_path,subfolder), logs_dispatch_after=logs_dispatch_after)
    gp2 = GamePlayer(game_state.duplicate(), agents[::-1])
    subfolder = "2_"+ gp2.players[0].name + "_vs_" + gp2.players[1].name
    gp2.play_games(games, random_seed = random_seed, logs = logs, logs_path=os.path.join(file_path,subfolder), logs_dispatch_after=logs_dispatch_after)
    #if file_path is not None:
    #    gp1.save_data(os.path.join(file_path , gp1.players[0].name + "_vs_" + gp1.players[1].name))
    #    gp2.save_data(os.path.join(file_path , gp2.players[0].name + "_vs_" + gp2.players[1].name))
    return gp1, gp2

def tree_data(node, divisions=1):
    """Division method Options: percentage, best_changed
        early_cut: only nodes that were added before a terminal state was reached by an expanded node"""

    nodes = node.subtree_nodes(node)
    n_nodes = len(nodes)
    id_block_step = n_nodes/divisions
    #print("n_nodes", n_nodes)
    #print("id_block_step", id_block_step)
    #print("max_id", max([x.expansion_index for x in nodes]))

    tree_data = pd.DataFrame()
    for node in nodes:
        node_df = pd.DataFrame(node.state.logs_data(), index=[0])
        node_df["id"] = node.expansion_index
        node_df["parent_id"] = node.parent.expansion_index if node.parent is not None else None
        for d in range(divisions):
            if d*id_block_step <= node.expansion_index and node.expansion_index <= (d+1)*id_block_step:
                node_df["id_block"] = d
                break
        #node_df["id_block"] = int((node.expansion_index/(n_nodes+1))/(1/divisions))
        node_df["visits"] = node.visits
        node_df["avg_reward"] = node.total_reward / node.visits if node.visits > 0 else np.nan
        tree_data = pd.concat([tree_data, node_df], ignore_index=True)

    return tree_data
    
def best_leaf_node(node, criteria="visits"):
    assert isinstance(node, mcts.Node), "node must be a mcts.Node"
    if node.state.is_terminal: return node
    else:
        max_value = -np.inf
        max_value_node = None
        for child in node.children.values():
            if criteria == "visits": node_value = child.visits
            elif criteria == "avg_reward": 
                if child.visits > 0:
                    node_value = child.total_reward / child.visits
                else: 
                    node_value = -np.inf
            if node_value > max_value:
                max_value = node_value
                max_value_node = child
        if max_value_node is None: 
            #print("max_value_node is None")
            return node
        return best_leaf_node(max_value_node, criteria=criteria)

def terminal_count(node):
    if node.state.is_terminal: return 1
    else:
        count = 0
        for child in node.children.values():
            count = count + terminal_count(child)
        return count

def mcts_decision_analysis(game_state, mcts_player, logs_path, runs = 1, random_seed = None): #Requires generalisation. Hardcoded for FOP
    "Runs a game with mcts_player and saves the tree data to logs_path"

    #Initialise logs
    tree_logs = pd.DataFrame()
    logs_by_run = pd.DataFrame()

    #Run game
    for i in range(runs):

        #Set random seed
        if random_seed is not None: 
            rd.seed(random_seed + i)
            np.random.seed(random_seed + i)
        else: 
            random_seed = rd.randint(0, 2**32)
            rd.seed(random_seed)
            np.random.seed(random_seed)

        action = mcts_player.choose_action(game_state)
        run_data = tree_data(mcts_player.root_node, divisions=3)

        #Add run data to logs
        run_data["run"] = [i for _ in range(len(run_data))]
        tree_logs = pd.concat([tree_logs, run_data], ignore_index=True)
        max_reward_leaf_node = best_leaf_node(mcts_player.root_node, "visits")
        max_visits_leaf_node = best_leaf_node(mcts_player.root_node, "avg_reward")
        this_run_data = {
            "run": i,
            "eval_of_max_reward_leaf_node": max_reward_leaf_node.state.function(max_reward_leaf_node.state.eval_point()),
            "eval_point_of_max_reward_leaf_node": str(max_reward_leaf_node.state.eval_point()),
            "eval_of_max_visits_leaf_node": max_visits_leaf_node.state.function(max_visits_leaf_node.state.eval_point()),
            "eval_point_of_max_visits_leaf_node": str(max_visits_leaf_node.state.eval_point()),
            "terminals_count": terminal_count(mcts_player.root_node),
            "agent_expanded_nodes": mcts_player.nodes_count-1,
            "nodes_by_iteration" : (mcts_player.nodes_count-1)/mcts_player.current_iterations if mcts_player.current_iterations > 0 else np.nan,
            "run_random_seed": random_seed
        }
        this_run_df = pd.DataFrame(this_run_data, index=[0])
        this_run_df = pd.concat([this_run_df, game_state.game_definition_data()], axis=1)
        this_run_df = pd.concat([this_run_df, mcts_player.choose_action_logs], axis=1)
        #this_run_df = pd.concat([this_run_df, mcts_player.root_node.node_data()], axis=1)
        logs_by_run = pd.concat([logs_by_run, this_run_df], ignore_index=True)
        mcts_player.dump_my_logs(os.path.join(logs_path, "Run_" + str(i)))
        #lm.dump_data(logs_by_run, file_path= os.path.join(logs_path, "Run_" + str(i)), file_name="logs_by_run.csv")

    #Update global logs
    global_data = {
        "avg_eval_of_max_reward_leaf_node" : st.mean(logs_by_run["eval_of_max_reward_leaf_node"]),
        "std_eval_of_max_reward_leaf_node" : st.stdev(logs_by_run["eval_of_max_reward_leaf_node"]) if len(logs_by_run) > 1 else np.nan,
        "avg_eval_of_max_visits_leaf_node" : st.mean(logs_by_run["eval_of_max_visits_leaf_node"]),
        "std_eval_of_max_visits_leaf_node" : st.stdev(logs_by_run["eval_of_max_visits_leaf_node"]) if len(logs_by_run) > 1 else np.nan,
        "avg_terminals_count" : st.mean(logs_by_run["terminals_count"]),
        "std_terminals_count" : st.stdev(logs_by_run["terminals_count"]) if len(logs_by_run) > 1 else np.nan,
        "avg_nodes_by_iteration" : st.mean(logs_by_run["nodes_by_iteration"]),
        "std_nodes_by_iteration" : st.stdev(logs_by_run["nodes_by_iteration"]) if len(logs_by_run) > 1 else np.nan,
        "avg_forward_model_calls" : st.mean(logs_by_run["forward_model_calls"]),
        "std_forward_model_calls" : st.stdev(logs_by_run["forward_model_calls"]) if len(logs_by_run) > 1 else np.nan,
        "avg_time" : st.mean(logs_by_run["current_time"]),
        "std_time" : st.stdev(logs_by_run["current_time"]) if len(logs_by_run) > 1 else np.nan,
        "avg_root_avg_reward": st.mean(logs_by_run["avg_reward"]),
        "std_root_avg_reward": st.stdev(logs_by_run["avg_reward"]) if len(logs_by_run) > 1 else np.nan,
        "random_seed": random_seed,
        "runs": runs,
    }
    global_data_df = pd.DataFrame(global_data, index=[0])
    global_data_df = pd.concat([global_data_df, game_state.game_definition_data()], axis=1)
    global_data_df = pd.concat([global_data_df, mcts_player.agent_data()], axis=1)

    #Dump logs
    lm.dump_data(global_data_df, file_path= logs_path, file_name="results.csv")
    lm.dump_data(logs_by_run, file_path= logs_path, file_name="logs_by_run.csv")
    lm.dump_data(tree_logs, file_path= logs_path, file_name="tree_data.csv")
    lm.dump_data(game_state.game_definition_data(), file_path= logs_path, file_name="game_definition_data.csv")
    return action

def evolved_formula_analysis(data):
   output={}
   terminals = ["n","N","Q"]
   assert terminals[-1] == "Q", "Q must be the last terminal"
   for terminal in terminals: #ensure Q is at the end, for compatibility
      terminal_count = 0
      all_appareances = []
      for f in data["evolved_formula"]:
         appearances = f.count(terminal)
         if terminal == "n":
            appearances -= f.count("ln")
         if appearances > 0:
            terminal_count += 1
         all_appareances.append(appearances)
      output["Avg_per_formula_" + terminal] = np.mean(all_appareances)
      output["Std_per_formula_" + terminal] = np.std(all_appareances)
      output["Appearance_rate_" + terminal] = terminal_count/len(data)

   trivial_stuck = 0 #no Q or n
   trivial_random = 0 #none
   trivial_greedy = 0 #Q no n
   for f in data["evolved_formula"]:
      if "Q" not in f:
         if f.count("n") - f.count("ln") < 1:
            trivial_random += 1
         else: trivial_stuck += 1
      else:
         if f.count("n") - f.count("ln") < 1:
            trivial_greedy += 1

   output["Trivial_stuck"] = trivial_stuck/len(data)
   output["Trivial_random"] = trivial_random/len(data)
   output["Trivial_greedy"] = trivial_greedy/len(data)
   output["No_change"] = len([x for x in data["evolved_formula_not_ucb1"] if not x])/len(data)
   #print(data["evolved_formula_not_ucb1"].unique())
   output["Non_Trivial"] = 1-output["Trivial_stuck"] - output["Trivial_random"] - output["Trivial_greedy"]

   output["Avg_nodes_" + terminal] = np.mean(data["evolved_formula_nodes"])
   output["Std_nodes_" + terminal] = np.std(data["evolved_formula_nodes"])
   output["Avg_depth_" + terminal] = np.mean(data["evolved_formula_depth"])
   output["Std_depth_" + terminal] = np.std(data["evolved_formula_depth"])

   output["More_than_10"] = len(data.loc[data["agent_expanded_nodes"] >= 0.1*data["max_iterations"]])/len(data)
   output["More_than_50"] = len(data.loc[data["agent_expanded_nodes"] >= 0.5*data["max_iterations"]])/len(data)
   output["Less_than_4_nodes_formula"] = len(data.loc[data["evolved_formula_nodes"] < 4])/len(data)
   


   output_df = pd.DataFrame({k:[v] for k,v in output.items()})
   return output_df

####Chess specific functions

def chess_puzzle_test(puzzle_row, agent, continue_moves = False, iterations_logs_step = 10, runs = 1):
    """
    Runs a puzzle test for a given agent and puzzle
    Input:
        puzzle_row: row from the puzzle database
        agent: mcts-based agent to test
        continue_moves: if True, the agent will play until the end of the puzzle. If False, the agent will only play the first move
        iterations_logs_step: number of iterations to run before collecting logs
        runs: number of runs to execute

    Output:
        Return detailed logs, experiment parameters logs
    """

    logs = pd.DataFrame()
    rd_squence = rd.sample(range(0, 2**32), runs)
    for run in range(runs):

        #Set random seed
        rd.seed(rd_squence[run])
        np.random.seed(rd_squence[run])

        #Setup state
        chess_state = chess_64.GameState()
        chess_state.set_puzzle_lichess_db(puzzle_row)

        #Get actions from the puzzle, minus the first one because it is already executed by set_puzzle_lichess_db
        remaining_moves = [m for m in puzzle_row["Moves"].split(" ")[1:]]
        corrects = [None for _ in remaining_moves]
        first_action_solution = None

        #The agent may play as white or black
        kickstart_mcts_agent(agent, chess_state)
        agent_player_index = chess_state.player_turn
        player_turn = chess_state.player_turn

        #Loop through the puzzle moves
        for move_idx, move in enumerate(remaining_moves):
            move = chess_state.board.parse_uci(move)
            print("On move " + str(move_idx) + "_" + str(move) + " of " + str(len(remaining_moves)) + " of puzzle " + str(puzzle_row["PuzzleId"]) + " of run " + str(run) + " of " + str(runs-1))
            if move_idx == 0: first_action_solution = str(move)

            #If it is the agent's turn, execute the action chosen by the agent
            if player_turn == agent_player_index:

                #Setup agent
                kickstart_mcts_agent(agent, chess_state)

                #chosen_action = agent.choose_action(chess_state)
                total_iterations = 0
                full_puzzle_expanded = False
                moves_from_solution_expanded = 0
                while total_iterations < agent.max_iterations:
                    for i in range(iterations_logs_step):
                        agent.iteration()
                    total_iterations += iterations_logs_step
                    chosen_action = agent.recommendation_policy()

                    #Update logs in the agent
                    agent.choose_action_logs = pd.DataFrame()
                    agent._update_choose_action_logs()

                    #Collect logs
                    move_data = {}
                    move_data["exp_run"] = [run]
                    move_data["run_seed"] = [rd_squence[run]]
                    move_data["puzzle_move_index"] = [move_idx]
                    move_data["iterations_executed"] = [total_iterations]
                    move_data["current_chosen_move"] = [str(chosen_action.move)]
                    move_data["would_return_correct_move"] = [str(chosen_action.move) == str(move)]
                    move_data["expected_move"] = [str(move)]

                    solution_action = chess_64.Action(move, player_turn=agent_player_index) #to get the dict key as used in the root_node
                    solution_expanded = True
                    if agent.root_node.can_be_expanded():
                        if not solution_action in list(agent.root_node.children.keys()):
                            solution_expanded = False
                    if solution_expanded:
                        move_data["solution_avg_reward"] = agent.root_node.children[solution_action].average_reward()
                        move_data["solution_visits"] = agent.root_node.children[solution_action].visits
                        move_data["solution_can_be_expanded"] = agent.root_node.children[solution_action].can_be_expanded()
                    else:
                        move_data["solution_avg_reward"] = None
                        move_data["solution_visits"] = None
                        move_data["solution_can_be_expanded"] = None

                    #Find children nodes with the highest average rewards and visits
                    children_nodes = [c for c in list(agent.root_node.children.values()) if c.edge_action != solution_action]
                    children_nodes.sort(key=lambda x: x.average_reward(), reverse=True)
                    move_data["nonsolution_highest_reward_avg_reward"] = children_nodes[0].average_reward()
                    move_data["nonsolution_highest_reward_visits"] = children_nodes[0].visits
                    move_data["nonsolution_highest_reward_move"] = str(children_nodes[0].edge_action)
                    move_data["solution_is_max_avg_reward"] = str(children_nodes[0].edge_action) == str(chosen_action.move)
                    #find reward ranking of the solution
                    solution_rank = 1
                    for c in children_nodes:
                        if c.edge_action == solution_action: break
                        solution_rank += 1
                    move_data["solution_reward_rank"] = solution_rank

                    children_nodes.sort(key=lambda x: x.visits, reverse=True)
                    move_data["nonsolution_most_visited_avg_reward"] = children_nodes[0].average_reward()
                    move_data["nonsolution_most_visited_visits"] = children_nodes[0].visits
                    move_data["nonsolution_most_visits_move"] = str(children_nodes[0].edge_action)
                    move_data["solution_is_max_visits"] = str(children_nodes[0].edge_action) == str(chosen_action.move)
                    solution_rank = 1
                    for c in children_nodes:
                        if c.edge_action == solution_action: break
                        solution_rank += 1
                    move_data["solution_visits_rank"] = solution_rank

                    move_data["nonsolutions_avg_avg_reward"] = st.mean([c.average_reward() for c in children_nodes])
                    move_data["nonsolutions_avg_stddev_reward"] = st.stdev([c.average_reward() for c in children_nodes]) if len(children_nodes) > 1 else np.nan
                    move_data["nonsolutions_avg_visits"] = st.mean([c.visits for c in children_nodes])
                    move_data["nonsolution_rewards_list"] = str([c.average_reward() for c in children_nodes])
                    move_data["nonsolution_visits_list"] = str([c.visits for c in children_nodes])
                    #See if the solution was expanded at any point
                    if not full_puzzle_expanded:
                        moves_from_solution_expanded = 0
                        if solution_expanded:
                            i_turn = 1 - agent_player_index
                            current_node = agent.root_node.children[solution_action]
                            full_puzzle_expanded = True
                            moves_from_solution_expanded = 1
                            for i_move in remaining_moves[move_idx+1:]:
                                imove_action = chess_64.Action(i_move, player_turn=i_turn)
                                if imove_action in list(agent.root_node.children.keys()):
                                    current_node = current_node.children[imove_action]
                                else:
                                    full_puzzle_expanded = False
                                    break
                                i_turn = 1 - i_turn
                                moves_from_solution_expanded += 1
                    else:
                        moves_from_solution_expanded = len(remaining_moves) - move_idx - 1
                    move_data["full_puzzle_expanded"] = full_puzzle_expanded
                    move_data["moves_from_solution_expanded"] = moves_from_solution_expanded
                    move_data["moves_from_solution_expanded_max"] = len(remaining_moves) - move_idx - 1


                    move_df = pd.DataFrame(move_data)
                    move_logs = pd.concat([move_df, agent.choose_action_logs], axis = 1)
                    move_logs.set_index(["exp_run", "puzzle_move_index", "iterations_executed"], inplace=True, append = True, drop = False)

                    #Find duplicated columns in move_logs
                    duplicated_columns = move_logs.columns[move_logs.columns.duplicated()]
                    if len(duplicated_columns) > 0:
                        move_logs.drop(columns=duplicated_columns, inplace=True)
                        #print("Dropped duplicated columns", duplicated_columns)

                    logs = pd.concat([logs, move_logs], axis = 0)

                #If the agent chose a different action, the puzzle is not solved
                if chosen_action.move != move:
                    corrects[move_idx] = False
                else:
                    corrects[move_idx] = True
                
                #If the agent chose the correct action, execute it
                chess_state.make_action(chess_64.Action(move, player_turn=chess_state.player_turn))
            
            #If not the agent's turn, execute the defensive move from the puzzle
            else:
                chess_state.make_action(chess_64.Action(move, player_turn=chess_state.player_turn))

            #Swap turn holder for the next move in the puzzle
            if continue_moves: player_turn = 1 - player_turn
            else: break

    experiment_logs = pd.DataFrame()
    experiment_logs = pd.concat([experiment_logs, agent.agent_data()], axis=1)
    puzzle_row_df = pd.DataFrame([puzzle_row.tolist()], columns=puzzle_row.index)
    experiment_logs = pd.concat([experiment_logs, puzzle_row_df], axis=1)
    experiment_logs["expected_move"] = [str(move)]
    correct_by_run = 0
    iteration_as_last_choice = []
    solution_rewards = []
    solution_visits = []
    best_nonsolution_rewards = []
    best_nonsolution_visits = []
    most_visited_nonsolution_rewards = []
    most_visited_nonsolution_visits = []
    avg_solution_reward_rank = []
    avg_solution_visits_rank = []
    avg_nonsolution_rewards = []
    avg_nonsolution_visits = []
    nonsolution_mostvisited_moves = []
    nonsolution_highest_reward_moves = []
    full_puzzle_expanded_list = []
    fm_calls_list = []
    for run in range(runs):
        run_logs = logs[logs["exp_run"]==run]
        #print("Initial_run_logs_len", str(len(run_logs)))
        final_row = run_logs.iloc[-1]

        if final_row["solution_avg_reward"] is not None:
            solution_rewards = solution_rewards + [final_row["solution_avg_reward"]]
            solution_visits = solution_visits + [final_row["solution_visits"]]
        best_nonsolution_rewards = best_nonsolution_rewards + [final_row["nonsolution_highest_reward_avg_reward"]]
        best_nonsolution_visits = best_nonsolution_visits + [final_row["nonsolution_highest_reward_visits"]]
        most_visited_nonsolution_rewards = most_visited_nonsolution_rewards + [final_row["nonsolution_most_visited_avg_reward"]]
        most_visited_nonsolution_visits = most_visited_nonsolution_visits + [final_row["nonsolution_most_visited_visits"]]
        avg_solution_reward_rank = avg_solution_reward_rank + [final_row["solution_reward_rank"]]
        avg_solution_visits_rank = avg_solution_visits_rank + [final_row["solution_visits_rank"]]
        avg_nonsolution_rewards = avg_nonsolution_rewards + [final_row["nonsolutions_avg_avg_reward"]]
        avg_nonsolution_visits = avg_nonsolution_visits + [final_row["nonsolutions_avg_visits"]]
        nonsolution_mostvisited_moves = nonsolution_mostvisited_moves + [final_row["nonsolution_most_visits_move"]]
        nonsolution_highest_reward_moves = nonsolution_highest_reward_moves + [final_row["nonsolution_highest_reward_move"]]
        full_puzzle_expanded_list = full_puzzle_expanded_list + [final_row["full_puzzle_expanded"]]
        fm_calls_list = fm_calls_list + [final_row["forward_model_calls"]]

        #if solved
        if final_row["current_chosen_move"] == first_action_solution:
            correct_by_run += 1
            #Find the last iteration where the agent chose the correct action without changing it again - Not working yet
            correct_run_logs = run_logs[run_logs["current_chosen_move"]==first_action_solution]
            last_unchanged_iteration = correct_run_logs["iterations_executed"].max()
            for i in reversed(range(len(correct_run_logs))[1:]):
                if abs(correct_run_logs.iloc[i]["iterations_executed"] - correct_run_logs.iloc[i-1]["iterations_executed"]) == iterations_logs_step:
                    last_unchanged_iteration = correct_run_logs.iloc[i-1]["iterations_executed"]
                else: 
                    break
            iteration_as_last_choice.append(last_unchanged_iteration)

    if len(iteration_as_last_choice) > 0:
        experiment_logs["iteration_solution_unchanged_when_solved_avg"] = st.mean(iteration_as_last_choice)
        if len(iteration_as_last_choice) > 1:
            experiment_logs["iteration_solution_unchanged_when_solved_stdev"] = st.stdev(iteration_as_last_choice)
        

    experiment_logs["solved_ratio"] = [correct_by_run/runs]
    experiment_logs["solution_avg_reward"] = [st.mean(solution_rewards)]
    experiment_logs["solution_avg_reward_stdev"] = [st.stdev(solution_rewards)] if len(solution_rewards) > 1 else [np.nan]
    experiment_logs["solution_visits"] = [st.mean(solution_visits)]
    experiment_logs["solution_visits_stdev"] = [st.stdev(solution_visits)] if len(solution_visits) > 1 else [np.nan]
    experiment_logs["nonsolution_highest_reward_avg_reward"] = [st.mean(best_nonsolution_rewards)]
    experiment_logs["nonsolution_highest_reward_avg_reward_stdev"] = [st.stdev(best_nonsolution_rewards)] if len(best_nonsolution_rewards) > 1 else [np.nan]
    experiment_logs["nonsolution_highest_reward_visits"] = [st.mean(best_nonsolution_visits)]
    experiment_logs["nonsolution_highest_reward_visits_stdev"] = [st.stdev(best_nonsolution_visits)] if len(best_nonsolution_visits) > 1 else [np.nan]
    experiment_logs["nonsolution_most_visited_avg_reward"] = [st.mean(most_visited_nonsolution_rewards)]
    experiment_logs["nonsolution_most_visited_avg_reward_stdev"] = [st.stdev(most_visited_nonsolution_rewards)] if len(most_visited_nonsolution_rewards) > 1 else [np.nan]
    experiment_logs["nonsolution_most_visited_visits"] = [st.mean(most_visited_nonsolution_visits)]
    experiment_logs["nonsolution_most_visited_visits_stdev"] = [st.stdev(most_visited_nonsolution_visits)] if len(most_visited_nonsolution_visits) > 1 else [np.nan]
    experiment_logs["solution_avg_reward_rank"] = [st.mean(avg_solution_reward_rank)]
    experiment_logs["solution_avg_reward_rank_stdev"] = [st.stdev(avg_solution_reward_rank)] if len(avg_solution_reward_rank) > 1 else [np.nan]
    experiment_logs["solution_visits_rank"] = [st.mean(avg_solution_visits_rank)]
    experiment_logs["solution_visits_rank_stdev"] = [st.stdev(avg_solution_visits_rank)] if len(avg_solution_visits_rank) > 1 else [np.nan]
    experiment_logs["nonsolutions_avg_avg_reward"] = [st.mean(avg_nonsolution_rewards)]
    experiment_logs["nonsolutions_avg_stdev_reward"] = [st.stdev(avg_nonsolution_rewards)] if len(avg_nonsolution_rewards) > 1 else [np.nan]
    experiment_logs["nonsolutions_avg_visits"] = [st.mean(avg_nonsolution_visits)]

    #find the most common nonsolution_most_visits_move
    experiment_logs["nonsolution_most_visits_move"] = [st.mode(nonsolution_mostvisited_moves)]
    experiment_logs["nonsolution_highest_reward_move"] = [st.mode(nonsolution_highest_reward_moves)]
    experiment_logs["full_puzzle_expanded"] = [st.mean([1 if x else 0 for x in full_puzzle_expanded_list])]
    experiment_logs["fm_calls"] = [st.mean(fm_calls_list)]
    experiment_logs["fm_calls_stdev"] = [st.stdev(fm_calls_list)] if len(fm_calls_list) > 1 else [np.nan]
    
    #Puzzle was solved
    return logs, experiment_logs

def kickstart_mcts_agent(agent, state):
    agent_previous_iterations = agent.max_iterations
    agent.max_iterations = 0
    agent.choose_action(state)
    agent.max_iterations = agent_previous_iterations
    return agent

def analyse_puzzles_by_tag(results_df):
    """
    results_df = dataframe with the results, edeom extended with moves count, depth, theme count
    file_name = for the output file
    """
    tag_dict = {}
    all_tags = ["ALLTOGETHER"]
    for row in results_df.iterrows():
        tags = row[1]['Themes'].split(' ')
        for tag in tags:
            if tag not in all_tags:
                all_tags.append(tag)

    for tag in all_tags:
        if tag == "ALLTOGETHER":
            subset_df = results_df
        else:
            subset_df = results_df[results_df['Themes'].str.contains(tag)]
            
        tag_count = len(subset_df)
        #correct_subset = subset_df[subset_df['solved_ratio'] == 1]
        #tag_correct_count = len(correct_subset)
        tag_dict[tag] = [tag_count/len(results_df),tag_count]
        #get average_rating
        if tag_count > 1:
            tag_dict[tag].append(subset_df['Rating'].mean())
            tag_dict[tag].append(subset_df['Rating'].std())
            tag_dict[tag].append(subset_df["Available_actions"].mean())
            tag_dict[tag].append(subset_df['Available_actions'].std())
            tag_dict[tag].append(subset_df["NbPlays"].mean())
            tag_dict[tag].append(subset_df['NbPlays'].std())
            tag_dict[tag].append(subset_df["Pieces"].mean())
            tag_dict[tag].append(subset_df['Pieces'].std())
            tag_dict[tag].append(subset_df["Theme_count"].mean())
            tag_dict[tag].append(subset_df['Theme_count'].std())
            tag_dict[tag].append(subset_df["Required_moves"].mean())
            tag_dict[tag].append(subset_df["Required_moves"].std())
        else:
            #print(subset_df["Rating"].iat[0])
            tag_dict[tag].append(subset_df['Rating'].iat[0])
            tag_dict[tag].append(0)
            tag_dict[tag].append(subset_df["Available_actions"].iat[0])
            tag_dict[tag].append(0)
            tag_dict[tag].append(subset_df["NbPlays"].iat[0])
            tag_dict[tag].append(0)
            tag_dict[tag].append(subset_df["Pieces"].iat[0])
            tag_dict[tag].append(0)
            tag_dict[tag].append(subset_df["Theme_count"].iat[0])
            tag_dict[tag].append(0)
            tag_dict[tag].append(subset_df["Required_moves"].iat[0])
            tag_dict[tag].append(0)

    tag_df = pd.DataFrame.from_dict(tag_dict, orient='index', columns=["appearance_ratio","count","average_rating","std_rating","average_available_moves","std_available_moves","average_nb_plays","std_nb_plays","pieces","std_pieces","tag_count","std_tag_count","required_moves","required_moves_std_depth"])
    return tag_df

def extend_lichess_db(file_path, file_name):
    """
    file_path = string path to file
    file_name = string name of file with .csv extension
    Add number of moves to puzzles
    Add moves depth to puzzles
    Add theme count to puzzles
    """
    game_state = chess_64.GameState()
    lichess_db = pd.read_csv(os.path.join(file_path, file_name))
    moves_count_list = [None for _ in range(len(lichess_db))]
    piece_count_list = [None for _ in range(len(lichess_db))]
    tags_count_list = [None for _ in range(len(lichess_db))]
    required_moves = [None for _ in range(len(lichess_db))]
    player_to_move = [None for _ in range(len(lichess_db))]

    for puzzle_idx, puzzle_row in lichess_db.iterrows():
        game_state.set_puzzle_lichess_db(puzzle_row)
        moves_count_list[puzzle_idx] = len(game_state.available_actions)


        fen = puzzle_row["FEN"].split(" ")[0]
        piece_count_list[puzzle_idx] = sum(c.isalpha() for c in fen)
        tags_count_list[puzzle_idx] = len(puzzle_row["Themes"].split(" "))
        required_moves[puzzle_idx] = int(len(puzzle_row["Moves"].split(" "))/2)
        player_to_move[puzzle_idx] = game_state.player_turn

        if puzzle_idx % 100000 == 0:
            print(puzzle_idx)

            #save in csv progress
            #moves_count_df = pd.DataFrame({"Moves":moves_count_list})
            #moves_count_df.to_csv("Datasets/lichess_db_puzzle_moves_temp.csv", index=False)

    lichess_db["Available_actions"] = moves_count_list
    lichess_db["Pieces"] = piece_count_list
    lichess_db["Theme_count"] = tags_count_list
    lichess_db["Required_moves"] = required_moves
    lichess_db["Player_to_move"] = player_to_move
    return lichess_db

def random_rollout(game_state, default_agent):
    """
    game_state = game state to rollout
    default_agent = agent to rollout with
    """
    logs_dict = {}
    list_action_count = []
    list_actions = []
    rollout_state = game_state.duplicate()
    start_time = time.time()
    while not rollout_state.is_terminal:
        action = default_agent.choose_action(rollout_state)
        list_actions.append(str(action))
        list_action_count.append(len(rollout_state.available_actions))
        rollout_state.make_action(action)
    logs_dict["time"] = time.time() - start_time
    logs_dict["turns"] = rollout_state.turn - game_state.turn
    logs_dict["winner"] = rollout_state.winner
    logs_dict["reward"] = rollout_state.reward[game_state.player_turn]
    logs_dict["actions"] = list_actions
    logs_dict["action_count"] = list_action_count
    logs_dict["average_action_count"] = np.mean(list_action_count)
    logs_dict["agent_name"] = default_agent.name
    logs_dict["final_state"] = rollout_state
    return logs_dict

#Visualisation
def fo_tree_histogram(data_list, function, title, divisions, n_buckets = 100, subplot_titles=None, max_x_location = None, y_ref_value = None, only_leafs = False):
    """
    Returns a figure with histogram for each agent
    Input: Array of dataframes, one for each agent.
    """

    if divisions in [2,3]: colors = ["#5B8C5A"
                ,"#56638A"
                , "#EC7316"]
    n_plots = len(data_list)
    even_spaces = 1/(n_plots+1)
    row_heights = [even_spaces for _ in range(n_plots)] + [even_spaces]
    fig = make_subplots(rows=n_plots+1
                        ,cols=1
                        ,shared_xaxes=True
                        ,vertical_spacing=0.04
                        ,row_heights=row_heights
                        ,subplot_titles = subplot_titles
                        , specs=[[{"secondary_y": True}] for _ in range(n_plots+1)]
                        ,x_title="Central point of the state represented by each node"
                        ,y_title='Total allocated nodes'
                        #,print_grid=True
                        )

    if function is not None:
        x = np.linspace(0.001,1,5000)
        y = [function([i]) for i in x]
        fig.add_trace(go.Scatter(x=x, y=y, showlegend=False,marker={"color":"#000000"}),row=1,col=1)

    show_legend = [True] + [False for _ in range(n_plots)]
    for i,data in enumerate(data_list):
        for div in range(divisions):

            #Set name of the divisions
            if div%divisions == 0: s1 = "{:2.0f}".format(100*(div/divisions))
            else: s1 = "{:2.1f}".format(100*(div/divisions))
            if (div+1)%divisions == 0: s2 = "{:2.0f}".format(100*((div+1)/divisions))
            else: s2 = "{:2.1f}".format(100*((div+1)/divisions))
            div_name = s1 + "% to " + s2 + "%"
            temp_data = data.loc[data["id_block"]==div]
            
            #Set number of buckets as per arguments
            if type(n_buckets) is list:
                if i < len(n_buckets):
                    n_bins = n_buckets[i]
                else: print("reached i ", i, " when max is ", str(len(n_buckets)))
            else: n_bins = n_buckets

            #Add histogram
            fig.add_trace(go.Histogram(x=temp_data.Eval_point_dimension0
                    , nbinsx=n_bins
                    , xbins={"start":0,"end":1,"size":1/n_bins}
                    , name = div_name
                    , showlegend=show_legend[i]
                    ,legendrank = 1000-div
                    ,histfunc = "avg"
                    #,histnorm="average"
                    , marker={"color":colors[div]})
            ,row=i+2,col=1)
            
            #Add secondary trace in the same plot
            #fig.add_trace(go.Scatter(x=x, y=y, showlegend=False,marker={"color":"black"}),row=i+1,col=1,secondary_y=True)

    fig.update_layout(margin=dict(l=70, r=10, t=30, b=50)
                        ,width=800
                        ,height=900
                        ,plot_bgcolor='rgba(0,0,0,0)'
                        #,plot_bgcolor="lightgray"
                        ,title={"text":title}
                        ,barmode='stack'
                        ,font = dict(family = "Arial", size = 14, color = "black")
                        ,legend=dict(
                            title = "Percentage of the total iterations"
                            ,orientation="h"
                            ,yanchor="top"
                            ,y=-0.075
                            #,xanchor="center"
                            #,x=0.4
                            ,bgcolor="white"#"lightgray"#"rgba(200, 255, 255, 0.8)"
                            ,font = dict(family = "Arial", size = 14, color = "black")
                            ,bordercolor="Black"
                            ,borderwidth=2
                            ,itemsizing='trace'
                            ,itemwidth = 30
                            ) 
                        )
    fig.update_xaxes(showline=True
                        , linewidth=2
                        , linecolor='black'
                        , mirror=True
                        )

    fig.update_yaxes(showline=True
                        ,mirror=True
                        , linewidth=2
                        , linecolor='black'
                        , nticks=5
                        #,tickmode = 'linear'
                        ,tick0 = 0
                        , gridcolor="#5B8C5A"
                        , gridwidth=0.1
                        #, dtick=5000
                        ,showgrid=False
                        )

    #Add line where the maximum x is
    list_of_shapes = []
    if max_x_location is not None:
        for subplot_n in range(1,n_plots+2):
            yref = "y" + str(subplot_n*2)
            list_of_shapes.append({'type': 'line','y0':0,'y1': 1,'x0':max_x_location, 
                                        'x1':max_x_location,'xref':"x",'yref':yref,
                                        'line': {'color': "#B10909",'width': 1.5, "dash":"dash"}})
    
    if y_ref_value is not None:
            for subplot_n in range(1,n_plots+1):
                yref = "y" + str(subplot_n*2-1)
                list_of_shapes.append({'type': 'line','y0':y_ref_value,'y1': y_ref_value,'x0':0, 
                                        'x1':1,'xref':"x",'yref':yref,
                                        'line': {'color': "#4F5D2F",'width': 1, "dash":"dash"}})
    
    if list_of_shapes != []:
        fig['layout'].update(shapes=list_of_shapes)
        for subplot_n in range(1, n_plots+2):
            yref = "y" + str(subplot_n)
            fig.add_shape(go.layout.Shape(type="line", yref=yref, xref="x", x0=max_x_location, x1 = max_x_location, y0=0, y1=1, line=dict(color="red", width=1),),row=subplot_n, col=1)
    
    for subplot_n in range(1,n_plots+2):
            fig['layout']['yaxis'+ str(subplot_n*2)]['visible'] = False

    return fig

def fo_tree_histogram_average(data_list, function, title, divisions, n_buckets = 100, subplot_titles=None, max_x_location = None, y_ref_value = None, height = 1000, width = 800, extra_bottom_margin = 0, legend_y = -0.075):
    """
    Returns a figure with histogram for each agent
    Input: Array of dataframes, one for each agent.
    """

    if divisions in [2,3]: 
        colors = ["#5B8C5A" ,"#56638A" , "#EC7316"]
        colors = ['#8cae8b', '#667295', '#8d450d']
    n_plots = len(data_list)
    if n_plots > 5: 
        even_spaces = 1/(n_plots+1)
        vertical_spacing = 0.04
    else: 
        even_spaces = 1/(n_plots+1)
        vertical_spacing = 0.1
    row_heights = [even_spaces for _ in range(n_plots)] + [even_spaces]

    #print("row_heights", row_heights)
    #print("subplot_titles", subplot_titles)
    #print("specs", [[{"secondary_y": True}] for _ in range(n_plots+1)])

    fig = make_subplots(rows=n_plots+1
                        ,cols=1
                        ,shared_xaxes=True
                        ,vertical_spacing=vertical_spacing
                        ,row_heights=row_heights
                        ,subplot_titles = subplot_titles
                        , specs=[[{"secondary_y": True}] for _ in range(n_plots+1)]
                        ,x_title="Central point of the state represented by each node"
                        ,y_title='Average allocated nodes'
                        #,print_grid=True
                        )

    if function is not None:
        x = np.linspace(0.001,1,5000)
        y = [function([i]) for i in x]
        fig.add_trace(go.Scatter(x=x, y=y, showlegend=False,marker={"color":"#000000"}),row=1,col=1)

    show_legend = [True] + [False for _ in range(n_plots)]

    processed_data = {}
    for i,data in enumerate(data_list):
        for div in range(divisions):

            #Set name of the divisions
            if div%divisions == 0: s1 = "{:2.0f}".format(100*(div/divisions))
            else: s1 = "{:2.1f}".format(100*(div/divisions))
            if (div+1)%divisions == 0: s2 = "{:2.0f}".format(100*((div+1)/divisions))
            else: s2 = "{:2.1f}".format(100*((div+1)/divisions))
            div_name = s1 + "% to " + s2 + "%"
            temp_data = data.loc[data["id_block"]==div]
            
            #Set number of buckets as per arguments
            if type(n_buckets) is list:
                if i < len(n_buckets):
                    n_bins = n_buckets[i]
                else: print("reached i ", i, " when max is ", str(len(n_buckets)))
            else: n_bins = n_buckets

            #Calc data
            temp_data_by_run = []#temp_data.groupby("run")
            for run in list(temp_data["run"].unique()):
                this_run_data = temp_data.loc[temp_data["run"]==run]
                temp_data_by_run.append( count_in_bins(this_run_data["Eval_point_dimension0"], n_bins))
            bar_data = {k:st.mean([d[k] for d in temp_data_by_run]) for k in temp_data_by_run[0].keys()}

            fig.add_trace(go.Bar(x=list(bar_data.keys()),
                                y=list(bar_data.values()),
                                offset=-1/(n_bins*2),
                                width=1/(n_bins),
                                name = div_name,
                                showlegend=show_legend[i],
                                legendrank = 1000-div,
                                marker={"color":colors[div]},
                                    ),row=i+2,col=1)

    fig.update_layout(margin=dict(l=70, r=10, t=30, b=50 + extra_bottom_margin)
                        ,width=width
                        ,height=height
                        ,plot_bgcolor='rgba(0,0,0,0)'
                        #,plot_bgcolor="lightgray"
                        ,title={"text":title}
                        ,barmode='stack'
                        ,font = dict(family = "Arial", size = 14, color = "black")
                        ,legend=dict(
                            title = "Percentage of the total iterations"
                            ,orientation="h"
                            ,yanchor="top"
                            ,y=legend_y
                            #,xanchor="center"
                            #,x=0.4
                            ,bgcolor="white"#"lightgray"#"rgba(200, 255, 255, 0.8)"
                            ,font = dict(family = "Arial", size = 14, color = "black")
                            ,bordercolor="Black"
                            ,borderwidth=2
                            ,itemsizing='trace'
                            ,itemwidth = 30
                            ) 
                        )
    fig.update_xaxes(showline=True
                        , linewidth=2
                        , linecolor='black'
                        , mirror=True
                        )

    fig.update_yaxes(showline=True
                        ,mirror=True
                        , linewidth=2
                        , linecolor='black'
                        , nticks=5
                        #,tickmode = 'linear'
                        ,tick0 = 0
                        , gridcolor="#5B8C5A"
                        , gridwidth=0.1
                        #, dtick=5000
                        ,showgrid=False
                        )

    #Add line where the maximum x is
    list_of_shapes = []
    if max_x_location is not None:
        for subplot_n in range(1,n_plots+2):
            yref = "y" + str(subplot_n*2)
            list_of_shapes.append({'type': 'line','y0':0,'y1': 1,'x0':max_x_location, 
                                        'x1':max_x_location,'xref':"x",'yref':yref,
                                        'line': {'color': "#B10909",'width': 1.5, "dash":"dash"}})
    
    if y_ref_value is not None:
            for subplot_n in range(1,n_plots+1):
                yref = "y" + str(subplot_n*2-1)
                list_of_shapes.append({'type': 'line','y0':y_ref_value,'y1': y_ref_value,'x0':0, 
                                        'x1':1,'xref':"x",'yref':yref,
                                        'line': {'color': "#4F5D2F",'width': 1, "dash":"dash"}})
    
    if list_of_shapes != []:
        fig['layout'].update(shapes=list_of_shapes)
        for subplot_n in range(1, n_plots+2):
            yref = "y" + str(subplot_n)
            fig.add_shape(go.layout.Shape(type="line", yref=yref, xref="x", x0=max_x_location, x1 = max_x_location, y0=0, y1=1, line=dict(color="red", width=1),),row=subplot_n, col=1)
    
    for subplot_n in range(1,n_plots+2):
            fig['layout']['yaxis'+ str(subplot_n*2)]['visible'] = False

    return fig

def fo_function_analysis(fo_state, title, max_depth=3, max_val=None):
   """
    Plots MCTS's fitness landscape for a 1d function
    
    Usage example:
    random_player = RandomPlayer()
    dummy_state = FunctionOptimisationState(players=[random_player], function=4, ranges=[[0,1]], splits=2)
    functions = dummy_state.function_list
    fig = fo_function_analysis(dummy_state, max_depth=5)
    fig.show()
   """

   #Initialize
   stop = fo_state.ranges[0][1]
   start = fo_state.ranges[0][0]

   #get max depth and step size
   step = fo_state.ranges[0][1] - fo_state.ranges[0][0]
   depth = 0
   while step > fo_state.minimum_step:
      depth += 1
      step = step / fo_state.splits
      if depth>1000:
         print("Infinite while error")
         break
   
   #Calculations
   bar_widths= []
   values_by_depth = {}
   x = np.linspace(start,stop,int((stop-start)/step))
   x=x[1:-1]
   y_dict = {xi:fo_state.function([xi]) for xi in x}
   for d in range(1,max_depth+1):
      bar_widths.append([1/(fo_state.splits**d) for _ in range(fo_state.splits**d)])
      divisions = fo_state.splits**d
      division_size = (stop-start)/divisions
      for i in range(divisions):
         section_begin = division_size*i
         section_end = division_size*(i+1)
         values = []
         for (k,v) in y_dict.items():
            if k > section_begin and k < section_end:
               values.append(v)
         values_by_depth[(d,i)] = stats.mean(values)
   #print(bar_widths)

   #create subplots
   n_plots = max_depth
   even_spaces = 1/(n_plots+1)
   row_heights = [even_spaces for _ in range(n_plots)] + [even_spaces]
   sub_titles = [""] + ["Tree Depth " + str(i+1) for i in range(max_depth)]
   fig = make_subplots(rows=n_plots+1, 
                       cols=1,
                       shared_xaxes=True,
                       vertical_spacing=0.04,
                       row_heights=row_heights, 
                       specs=[[{"secondary_y": True}] for _ in range(n_plots+1)], 
                       subplot_titles=sub_titles)
   
   #add function plot
   x = np.linspace(0.001,1,5000)
   y = [fo_state.function([i]) for i in x]
   fig.add_trace(go.Scatter(x=x, 
                            y=y, 
                            showlegend=False,
                            marker={"color":"black"}),
                  row=1,col=1)
   if max_val is not None:
         fig.add_trace(go.Scatter(x=[max_val,max_val], 
                                  y=[0,1], 
                                  line=dict(color="#B10909", width=2, dash='dash'),
                                  showlegend=True,
                                  marker={"color":"#B10909"},
                                  name="Localisation of the global maximum of the function"),
                        row=1,col=1)

   #add analysis plots
   for d in range(1,max_depth+1):
      show_legend = False
      if d == max_depth:
         show_legend = True
      widths = bar_widths[d-1]
      x = np.linspace(start,stop,((fo_state.splits**(d))*2)+1)
      x = [x[i] for i in range(1,len(x)) if i%2]
      x=np.cumsum(widths)-widths
      valid_keys = [k for k in values_by_depth.keys() if k[0] == d]
      y = [values_by_depth[k] for k in valid_keys]

      #Find max and change color
      max_y=max(y)
      colors = ["#5B8C5A" for _ in range(len(y))]
      #colors = ["#56638A" for _ in range(len(y))]
      textures = ["" for _ in range(len(y))]
      legend_groups = ["any" for _ in range(len(y))]
      max_texture = "x"
      #max_color = "#56638A"
      max_color = "#303C64"
      for i, y_i in enumerate(y):
         if abs(max_y-y_i) < 0.0001:
            colors[i] = max_color
            #colors[i] = "#FC738C"
            textures[i] = max_texture
            legend_groups[i] = "max"

      solidity = 0.4
      fig.add_trace(go.Bar(x=x
                           , y=y
                           , showlegend=show_legend
                           ,marker_color=colors
                           ,width=widths
                           ,offset=0
                           ,marker_pattern_shape=textures
                           ,name="Initial belief value for each node"
                           ,marker= dict(pattern = dict(solidity=solidity))
                           ,legendrank=1002
                           )
                        ,row=d+1,col=1)
      fig.add_trace(go.Bar(x=x
                           , y=[0 for _ in range(len(y))]
                           , showlegend=show_legend
                           #,marker_color="#FC738C"#"#56638A"
                           ,marker_color=max_color
                           ,width=widths
                           ,offset=0
                           ,marker_pattern_shape=max_texture
                           ,name="Maximum initial belief value at this depth"
                           ,legendrank=1001
                           ,marker= dict(pattern = dict(solidity=solidity))
                           )
                        ,row=d+1,col=1)
      fig.update_layout(legend= {'itemsizing': 'constant'})

      #fig.update_traces(marker_pattern_shape=textures)
      fig.add_trace(go.Scatter(x=[start,stop], y=[max(y),max(y)], line=dict(color=max_color, width=2, dash='dot'),showlegend=show_legend,marker={"color":"#56638A"},
                               name="Comparison of the maximum initial belief value available at this depth"),row=d+1,col=1)
      if max_val is not None:
         fig.add_trace(go.Scatter(x=[max_val,max_val], y=[0,1], line=dict(color="#B10909", width=2, dash='dash'),showlegend=False,marker={"color":"#B10909"}),row=d+1,col=1)

      #add vertical lines
      if d > 1:
         x_breaks = np.linspace(start,stop,fo_state.splits**(d-1)+1)
         x_breaks = x_breaks[1:-1]
         for x_break in x_breaks:
            fig.add_trace(go.Scatter(x=[x_break,x_break], y=[0,1], showlegend=False,marker={"color":"black"}),row=d+1,col=1)
            pass

   #update fig layout
   fig.update_layout(barmode='stack')
   fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                     width=800,
                     height=800,
                     plot_bgcolor='rgba(0,0,0,0)',
                     title={"text": title},
                     font = dict(family = "Arial", size = 14, color = "black")
                ,legend=dict(
                    #title = "Formula",
                    orientation="h",
                    yanchor="top",
                    y=-0.035,
                    xanchor="center",
                    x=0.5,  
                    font = dict(family = "Arial", size = 14, color = "black"),
                    #bordercolor="LightSteelBlue",
                    borderwidth=2,
                    itemsizing='trace',
                    itemwidth = 30
                    )  )
   #fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='black')
   fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
   fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
   # fig.update_xaxes(range=[start,stop])
   fig.update_xaxes(range=[0,1])
   fig.update_yaxes(range=[0,1])
   if max_val is not None and False:
         print(max_val)
         fig.add_shape({'type': 'line','y0':0,'y1': 1,'x0':max_val, 
                        'x1':max_val,
                        'line': {'color': "#B10909",
                                 'width': 1.5, 
                                 "dash":"dash",
                                 }}, row="all", col=1)
   return fig

def interpolate_colors(color1, color2, factor=0.5):
    return tuple(int(a + (b - a) * factor) for a, b in zip(color1, color2))

def darken_color(color, factor=0.5):
    return interpolate_colors(color, (0, 0, 0), factor)

def brighten_color(color, factor=0.5):
    return interpolate_colors(color, (255, 255, 255), factor)

def color_hex_to_rgb(color):
    color = color.lstrip('#')
    return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))

def color_rgb_to_hex(color):
    return '#%02x%02x%02x' % color

def view_color_palette(color_palette):
    """ Color palette is a list of hex colors"""
    fig, ax = plt.subplots(1, len(color_palette), figsize=(len(color_palette)*2, 2))

    for i, color in enumerate(color_palette):
        ax[i].add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        ax[i].text(0.5, -0.2, str(color), ha='center', va='center')
        ax[i].axis('off')

    plt.show()

def count_in_bins(x, n_bins, start=0, stop=1):
    assert n_bins > 1, "n_bins must be greater than 1"
    breaks = np.linspace(start=start, stop=stop, num=n_bins+1)
    half_step = (breaks[1]-breaks[0])/2
    counts = {i:0 for i in breaks[:-1]}
    for xi in x:
        for i in range(len(breaks)-1):
            if xi >= breaks[i] and xi < breaks[i+1]:
                counts[breaks[i]] += 1
                break
    count_data = {k+half_step:counts[k] for k in counts.keys()}
    return count_data

def generate_color_palette(num_colors):
    palette = []
    for i in range(num_colors):
        hue = i / num_colors
        # Choosing a fixed saturation and lightness to ensure high contrast
        saturation, lightness = 0.8, 0.45
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        palette.append(hex_color)
    return palette