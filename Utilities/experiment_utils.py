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


import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

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
        safe_count = 0
        while not gs.is_terminal:
            start_time = time.time()
            action = self.players[gs.player_turn].choose_action(gs)
            selection_time = time.time() - start_time

            #Update logs
            if logs:
                action_log = self.players[gs.player_turn].choose_action_logs #Assumes this log is single row
                action_log["game_index"] = self.games_count
                action_log["selection_time"] = selection_time
                action_log["returned_action"] = str(action)
                action_log["pg_player"] = str(self.players[gs.player_turn])
                action_logs = pd.concat([action_logs, action_log], ignore_index=True)
                game_logs = pd.concat([game_logs, gs.logs_data()], ignore_index=True)

            #safety check|            
            safe_count += 1
            if safe_count > 1000: 
                print("Safe count exceeded")
                print(gs.logs_data())
                print("Last action:" + str(action))
                break
            assert safe_count < 1000, "Safe count exceeded"

            #Make action
            gs.make_action(action)

        if logs:
            #Final logs by action
            final_logs_by_action = pd.concat([action_logs, game_logs], axis=1)


            #Final logs by game
            final_logs_by_game_dict = {}
            for i, player in enumerate(self.players):
                final_logs_by_game_dict["Player_" + str(i)] = str(player)
            final_logs_by_game_dict["Game_random_seed"] = random_seed
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

    def play_games(self, n_games, random_seed = None, logs = True, logs_dispatch_after = 10, logs_path = None) -> None:
        "Plays n_games games"
        if random_seed is not None: rd.seed(random_seed)
        else: rd.seed(rd.randint(0, 2**32))
        seeds = rd.sample(range(0, 2**32), n_games)
        for i in range(n_games):
            if logs: print("Playing game " + str(i) + " of " + str(n_games) + ", agents: " + str([a.name for a in self.players]))
            self.play_game(random_seed = seeds[i], logs = logs)
            if logs:
                if i % logs_dispatch_after == 0:
                    self.save_data(logs_path)
        if logs: self.save_data(logs_path)

    def save_data(self, file_path):
        "Saves logs to file_path"
        lm.dump_data(self.logs_by_game, file_path= file_path, file_name="logs_by_game.csv")
        lm.dump_data(self.logs_by_action, file_path= file_path, file_name="logs_by_action.csv")
        lm.dump_data(self.results(), file_path= file_path, file_name="results.csv")
        for i,p in enumerate(self.players):
            lm.dump_data(p.agent_data(), file_path= file_path, file_name="p" + str(i) + "_data.csv")

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
#
# 
def play_match(agents, game_state, games, file_path = None, random_seed = None, logs = True, logs_dispatch_after = 10) -> None:
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

    #Set random seed
    if random_seed is not None: rd.seed(random_seed)
    else: 
        random_seed = rd.randint(0, 2**32)
        rd.seed(random_seed)

    #Initialise logs
    tree_logs = pd.DataFrame()
    logs_by_run = pd.DataFrame()

    #Run game
    for i in range(runs):
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
        }
        this_run_df = pd.DataFrame(this_run_data, index=[0])
        this_run_df = pd.concat([this_run_df, game_state.game_definition_data()], axis=1)
        this_run_df = pd.concat([this_run_df, mcts_player.choose_action_logs], axis=1)
        this_run_df = pd.concat([this_run_df, mcts_player.root_node.node_data()], axis=1)
        logs_by_run = pd.concat([logs_by_run, this_run_df], ignore_index=True)
        mcts_player.dump_my_logs(os.path.join(logs_path, "Run_" + str(i)))

    #Update global logs
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
    global_data_df = pd.concat([global_data_df, game_state.game_definition_data()], axis=1)
    global_data_df = pd.concat([global_data_df, mcts_player.agent_data()], axis=1)

    #Dump logs
    lm.dump_data(global_data_df, file_path= logs_path, file_name="results.csv")
    lm.dump_data(logs_by_run, file_path= logs_path, file_name="logs_by_run.csv")
    lm.dump_data(tree_logs, file_path= logs_path, file_name="tree_data.csv")
    lm.dump_data(game_state.game_definition_data(), file_path= logs_path, file_name="game_definition_data.csv")
    return action


#Visualisation
def fo_tree_histogram(data_list, function, title, divisions, n_buckets = 100, subplot_titles=None, max_x_location = None, y_ref_value = None):
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

def fo_tree_histogram_average(data_list, function, title, divisions, n_buckets = 100, subplot_titles=None, max_x_location = None, y_ref_value = None):
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