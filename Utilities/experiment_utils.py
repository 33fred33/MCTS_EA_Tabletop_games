import cProfile
import random as rd
import time
from collections import defaultdict
import numpy as np
from itertools import cycle
import pandas as pd

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
        for i in range(len(self.players)):
            results["Player_" + str(i)] = str(self.players[i])
            results["win rate_" + str(i)] = self.win_count[i] / self.games_count
        results["draw rate"] = self.win_count["Draw"] / self.games_count
        results = pd.DataFrame(results, index=[0])
        return results

#def play_match()

def tree_data(node, divisions=1):
    """Division method Options: percentage, best_changed
        early_cut: only nodes that were added before a terminal state was reached by an expanded node"""

    nodes = tree_nodes(node)
    n_nodes = len(nodes)
    id_block_step = n_nodes/divisions
    print("tree_nodes", str([str(x) for x in nodes]))
    print("id_block_step", id_block_step)
    print("max_id", max([x.expansion_index for x in nodes]))

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

def tree_nodes(node):
    if len(node.children) == 0:
        return [node]
    else:
        nodes = [node]
        for child in node.children.values():
            nodes = nodes + tree_nodes(child)
        return nodes
    
def mcts_decision_analysis(game_state, mcts_player, logs_path, runs = 1, random_seed = None):
    "Runs a game with mcts_player and saves the tree data to logs_path"
    if random_seed is not None: rd.seed(random_seed)
    else: rd.seed(rd.randint(0, 2**32))
    tree_logs = pd.DataFrame()
    for i in range(runs):
        action = mcts_player.choose_action(game_state)
        run_data = tree_data(mcts_player.root_node, divisions=3)
        run_data["run"] = [i for _ in range(len(run_data))]
        tree_logs = pd.concat([tree_logs, run_data], ignore_index=True)
    lm.dump_data(tree_logs, file_path= logs_path, file_name="tree_data.csv")
    lm.dump_data(mcts_player.agent_data(), file_path= logs_path, file_name="agent_data.csv")
    return action