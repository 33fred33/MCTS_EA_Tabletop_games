import random as rd
from Agents.base_agents import BaseAgent
import pandas as pd
import Utilities.logs_management as lm
import numpy as np
import time
from Agents.vanilla_mcts import Node

class Minimax_Node(Node):
    def __init__(self, state=None, parent=None, edge_action=None, expansion_index=None, is_chance_node=False, random_event_type=None):
        super().__init__(state, parent, edge_action, expansion_index, is_chance_node, random_event_type)
        self.value = None
        self.depth = None
class Minimax(BaseAgent):
    
    player : int = 0

    def __init__(self, heuristic_function=None, name="Minimax", logs = False, pruning = True, probing_factor = 1, move_ordering_function = None, max_depth = np.inf, star_L = 0, star_U = 1, max_time = np.inf, max_fm = np.inf):
        """
        
        star_L: lower bound for the *-minimax family of algorithms
        star_U: upper bound for the *-minimax family of algorithms
        probing_factor: relevant for *-minimax family of algorithms
        heuristic_function: function that takes a state and a player and returns a value
        """
        self.name = name
        self.logs = logs
        self.choose_action_logs = pd.DataFrame({"Player":self.name, "player_name":self.name}, index=[0])
        self.isAIPlayer = True
        self.current_fm = 0
        self.pruning = pruning
        self.probing_factor = probing_factor #relevant for *-minimax family of algorithms
        self.move_ordering_function = move_ordering_function
        self.max_depth = max_depth
        self.max_time = max_time
        self.max_fm = max_fm
        self.star_L = star_L
        self.star_U = star_U
        if heuristic_function is None:
            def heuristic_function(state, player): #default for carcassonne
                if state.is_terminal:
                    to_return = state.reward[player]
                else:
                    to_return = state.Scores[player+2]/state.max_possible_score
                assert to_return is not None, "Heuristic function returned None in state: " + str(state) + " and player: " + str(player)
                return to_return
        self.heuristic_function = heuristic_function
    
    def choose_action(self, state, verbose=False):
        #Finds immediate wins, and returns that action. Random otherwise.
        self.current_fm = 0
        self.current_time = 0
        self.start_time = time.time()
        self.player = state.player_turn

        #create tree
        self.nodes_count = 0
        self.root_node = Minimax_Node(state=state, expansion_index=self.nodes_count)
        self.nodes_count += 1

        #Print warnings
        if state.players == 2 and self.star_L == 0:
            print("WARNING: star_L = 0 might be wrong for games with more than one player")
        elif state.players == 1 and self.star_L != 0:
            print("WARNING: star_L != 0 might be wrong for games with one player")

        self.action_values = {action:None for action in state.available_actions}
        for action in state.available_actions:
            duplicate_state = state.duplicate()
            duplicate_state.make_action(action)
            self.current_fm += 1
            if duplicate_state.random_events != []:
                #print("Minimax agent found random events. Using expectimax")
                self.action_values[action] = self.expectimax_search(state = duplicate_state, depth = 1, verbose=verbose)
            else:
                self.action_values[action] = self.minimax_search(state = duplicate_state, depth = 1, verbose=verbose)
            
            
        #get maximum value and corresponding action
        if verbose:
            print("Minimax agent values: ")
            #sort action_values by value
            self.action_values = {k: v for k, v in sorted(self.action_values.items(), key=lambda item: item[1], reverse=True)}
            for action_name, value in self.action_values.items():
                print("{:.3f}".format(value), ":", action_name)
        max_value = max(self.action_values.values())
        to_return = rd.choice([action for action, value in self.action_values.items() if value == max_value])

        #Update logs
        if self.logs:
            self._update_choose_action_logs()
            self.choose_action_logs["chosen_action"] = [to_return]
            self.choose_action_logs["max_value"] = [max_value]
            self.choose_action_logs["values"] = [str({str(a):v for a,v in self.action_values.items()})]

        return to_return

    def minimax_search(self, depth = 0, state = None, alpha = -np.inf, beta = np.inf, node = None, expectimax=False, verbose=False):
        if depth >= self.max_depth or state.is_terminal or self.stopping_criteria():
            return self.heuristic_function(state, self.player)
        if state.player_turn == self.player:
            max_value = -np.inf
            for action in state.available_actions:
                duplicate_state = state.duplicate()
                duplicate_state.make_action(action)
                self.current_fm += 1
                if expectimax: #expectimax
                    max_value = max(max_value, self.expectimax_search(depth + 1, duplicate_state, alpha, beta))
                else:
                    max_value = max(max_value, self.minimax_search(depth + 1, duplicate_state, alpha, beta))
                alpha = max(alpha, max_value)
                if self.pruning and alpha >= beta or self.stopping_criteria():
                    break
            return max_value
        else:
            min_value = np.inf
            for action in state.available_actions:
                duplicate_state = state.duplicate()
                duplicate_state.make_action(action)
                self.current_fm += 1
                if expectimax:
                    min_value = min(min_value, self.expectimax_search(depth + 1, duplicate_state, alpha, beta))
                else:
                    min_value = min(min_value, self.minimax_search(depth + 1, duplicate_state, alpha, beta))
                beta = min(beta, min_value)
                if self.pruning and alpha >= beta:
                    break
            return min_value

    def stopping_criteria(self):
        if self.current_fm >= self.max_fm:
            return True
        if self.current_time >= self.max_time:
            return True
        return False

    def expectimax_search(self, depth = 0, state = None, alpha = -np.inf, beta = np.inf, node = None, verbose=False):
        if state.is_terminal or depth >= self.max_depth or self.stopping_criteria():
            return self.heuristic_function(state, self.player)
        cumulative_score = 0
        random_event_probabilities = state.random_event_probabilities()
        #if verbose:
        #    print("random_event_probabilities: ", random_event_probabilities)
        for outcome, probability in random_event_probabilities.items():
            #print("outcome: ", outcome, "probability: ", probability)
            duplicate_state = state.duplicate()
            duplicate_state.sample_random_event(enforced_outcome = outcome)
            #if verbose:
            #    print("Trying outcome: ", outcome, "with probability: ", probability, "with available moves:", len(duplicate_state.available_actions))

            #check state is not terminal. In carcassonne, sampling a random event can lead to a terminal state if the tile is burned
            if duplicate_state.is_terminal:
                cumulative_score += probability*self.heuristic_function(duplicate_state, self.player)
            else:
                cumulative_score += probability*self.minimax_search(depth, duplicate_state, alpha, beta, expectimax=True)
        return cumulative_score

        
    def _update_choose_action_logs(self):
        data_dict = {
            "forward_model_calls": self.current_fm,
            "playing_as": self.player,
            "current_time": time.time() - self.start_time,
        }
        action_df = pd.DataFrame(data_dict, index=[0])
        action_df = pd.concat([action_df, self.agent_data()], axis=1)
        self.choose_action_logs = action_df

    def agent_data(self):
        return pd.DataFrame({"Player":str(self), 
                             "player_name":self.name, 
                             "probing_factor":self.probing_factor, 
                             "pruning":self.pruning,
                             "max_depth":self.max_depth,
                                "max_time":self.max_time,
                                "max_fm":self.max_fm,
                                "star_L":self.star_L,
                                "star_U":self.star_U,
                                "heuristic_function":self.heuristic_function.__name__,
                             },
                               index=[0])
    
    def dump_my_logs(self, file_path):
        for logs in [self.agent_data(), self.choose_action_logs]:
            lm.dump_data(logs, file_path, "agent_data.csv")
            lm.dump_data(self.choose_action_logs, file_path, "choose_action_logs.csv")

    def __str__(self):
        return self.name