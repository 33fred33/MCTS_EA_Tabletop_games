import cProfile
import random as rd
import time
from collections import defaultdict
import numpy as np
from itertools import cycle
import pandas as pd
import statistics as st

import Games.mnk as mnk
import Agents.random as arand
import Agents.vanilla_mcts as mcts 
import os
import Utilities.logs_management as lm

class GamePlayer():
    def __init__(self, game_state, players) -> None:
        self.game_state = game_state
        self.players = players
        self.games_count = 0
        self.logs_by_game = pd.DataFrame()
        self.logs_by_action = pd.DataFrame()
        self.win_count = {i:0 for i in range(len(players))}
        self.win_count["Draw"] = 0
        
    def play_game(self, random_seed = None, logs = True):
        "Plays a game. If logs is true, adds data to the class logs"

        gs = self.game_state.duplicate()

        #Set random seed
        if random_seed is not None: rd.seed(random_seed)
        else: 
            random_seed = rd.randint(0, 2**32)
            rd.seed(random_seed)

        #Set logs
        action_logs = pd.DataFrame()
        game_logs = pd.DataFrame()
        if logs:
            for p in self.players:
                p.logs = True

        #Play game
        while not gs.is_terminal:
            start_time = time.time()
            action = self.players[gs.player_turn].choose_action(gs)
            selection_time = time.time() - start_time

            #Update logs
            if logs:
                #ref: https://stackoverflow.com/questions/24284342/insert-a-row-to-pandas-dataframe
                #action_logs = pd.concat([action_logs, pd.DataFrame([[
                #    str(self.players[gs.player_turn]),       #player
                #    str(action),                        #"chosen_action"
                #    selection_time,                      #"time"
                #    self.games_count                   #"game_index"
                #]], columns=action_logs.columns)], ignore_index=True)
                action_log = self.players[gs.player_turn].choose_action_logs #Assumes this log is single row
                action_log["game_index"] = self.games_count
                action_log["selection_time"] = selection_time
                action_log["returned_action"] = str(action)
                action_log["pg_player"] = str(self.players[gs.player_turn])
                #action
                action_logs = pd.concat([action_logs, action_log], ignore_index=True)
                game_logs = pd.concat([game_logs, gs.logs_data()], ignore_index=True)

            #Make action    
            gs.make_action(action)

        if logs:
            #Final logs by action
            final_logs_by_action = pd.concat([action_logs, game_logs], axis=1)

            #Final logs by game
            final_logs_by_game_dict = {}
            for i, player in enumerate(self.players):
                final_logs_by_game_dict["Player_" + str(i)] = str(player)
            final_logs_by_game_dict["Random_seed"] = random_seed
            final_logs_by_game_dict["Game_index"] = self.games_count
            final_logs_by_game = pd.DataFrame(final_logs_by_game_dict, index=[0])
            final_logs_by_game = pd.concat([final_logs_by_game, gs.logs_data()], axis=1)
            
            #Update class logs
            self._update_logs_by_game(final_logs_by_game)
            self._update_logs_by_action(final_logs_by_action)
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

    def play_games(self, n_games, random_seed = None, logs = True):
        "Plays n_games games"
        if random_seed is not None: rd.seed(random_seed)
        else: rd.seed(rd.randint(0, 2**32))
        seeds = rd.sample(range(0, 2**32), n_games)
        for i in range(n_games):
            self.play_game(random_seed = seeds[i], logs = logs)

    def save_data(self, file_path):
        "Saves logs to file_path"
        lm.dump_data(self.logs_by_game, file_path= file_path, file_name="logs_by_game.csv")
        lm.dump_data(self.logs_by_action, file_path= file_path, file_name="logs_by_action.csv")
        lm.dump_data(self.results(), file_path= file_path, file_name="results.csv")
        for i,p in enumerate(self.players):
            lm.dump_data(p.agent_data(), file_path= file_path, file_name="p" + str(i) + "_data.csv")

        #self.logs_by_game.to_csv(file_path + "_by_game.csv", mode="a", header = not os.path.exists(file_path))
        #self.logs_by_action.to_csv(file_path + "_by_action.csv", mode="a", header = not os.path.exists(file_path))
        #self.results().to_csv(file_path + "_results.csv", mode="a", header = not os.path.exists(file_path))
        #, mode="a", header = not os.path.exists(file_path))

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
#def play_match()

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
    if random_seed is not None: rd.seed(random_seed)
    else: 
        random_seed = rd.randint(0, 2**32)
        rd.seed(random_seed)
    tree_logs = pd.DataFrame()
    logs_by_run = pd.DataFrame()
    for i in range(runs):
        action = mcts_player.choose_action(game_state)
        run_data = tree_data(mcts_player.root_node, divisions=3)
        run_data["run"] = [i for _ in range(len(run_data))]
        tree_logs = pd.concat([tree_logs, run_data], ignore_index=True)

        max_reward_leaf_node = best_leaf_node(mcts_player.root_node, "visits")
        max_visits_leaf_node = best_leaf_node(mcts_player.root_node, "avg_reward")
        this_run_data = {
            "run": i,
            "eval_of_max_reward_leaf_node": game_state.function(max_reward_leaf_node.state.eval_point()),
            "eval_point_of_max_reward_leaf_node": str(max_reward_leaf_node.state.eval_point()),
            "eval_of_max_visits_leaf_node": game_state.function(max_reward_leaf_node.state.eval_point()),
            "eval_point_of_max_visits_leaf_node": str(max_visits_leaf_node.state.eval_point()),
            "terminals_count": terminal_count(mcts_player.root_node),
            "agent_expanded_nodes": mcts_player.nodes_count-1,
            "nodes_by_iteration" : (mcts_player.nodes_count-1)/mcts_player.current_iterations if mcts_player.current_iterations > 0 else np.nan,
        }
        this_run_df = pd.DataFrame(this_run_data, index=[0])
        this_run_df = pd.concat([this_run_df, game_state.game_definition_data()], axis=1)
        this_run_df = pd.concat([this_run_df, mcts_player.choose_action_logs], axis=1)
        this_run_df = pd.concat([this_run_df, mcts_player.root_node.node_data()], axis=1)
        logs_by_run = pd.concat([logs_by_run, this_run_df], ignore_index=True)
    global_data = {
        "avg_eval_of_max_reward_leaf_node" : st.mean(logs_by_run["eval_of_max_reward_leaf_node"]),
        "std_eval_of_max_reward_leaf_node" : st.stdev(logs_by_run["eval_of_max_reward_leaf_node"]),
        "avg_eval_of_max_visits_leaf_node" : st.mean(logs_by_run["eval_of_max_visits_leaf_node"]),
        "std_eval_of_max_visits_leaf_node" : st.stdev(logs_by_run["eval_of_max_visits_leaf_node"]),
        "avg_terminals_count" : st.mean(logs_by_run["terminals_count"]),
        "std_terminals_count" : st.stdev(logs_by_run["terminals_count"]),
        "avg_nodes_by_iteration" : st.mean(logs_by_run["nodes_by_iteration"]),
        "std_nodes_by_iteration" : st.stdev(logs_by_run["nodes_by_iteration"]),
        "avg_forward_model_calls" : st.mean(logs_by_run["forward_model_calls"]),
        "std_forward_model_calls" : st.stdev(logs_by_run["forward_model_calls"]),
        "avg_time" : st.mean(logs_by_run["current_time"]),
        "std_time" : st.stdev(logs_by_run["current_time"]),
        "avg_root_avg_reward": st.mean(logs_by_run["avg_reward"]),
        "std_root_avg_reward": st.stdev(logs_by_run["avg_reward"]),
        "random_seed": random_seed,
        "runs": runs,
    }
    global_data_df = pd.DataFrame(global_data, index=[0])
    lm.dump_data(global_data_df, file_path= logs_path, file_name="results.csv")
    lm.dump_data(logs_by_run, file_path= logs_path, file_name="logs_by_run.csv")
    lm.dump_data(tree_logs, file_path= logs_path, file_name="tree_data.csv")
    lm.dump_data(mcts_player.agent_data(), file_path= logs_path, file_name="agent_data.csv")
    return action