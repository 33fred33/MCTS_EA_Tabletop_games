import operator
from operator import attrgetter
import numpy as np
import time
import random as rd
from statistics import mean
import math
import os
import pandas as pd
from Agents.vanilla_mcts import MCTS_Player, Node
from Agents.random import RandomPlayer
import statistics as st
import Utilities.logs_management as lm
import Utilities.experiment_utils as eu
import Games.base_games as bg


class MCTS_Solver(MCTS_Player):

    def __init__(self, 
                 rollouts=1, 
                 c=math.sqrt(2), 
                 max_fm=np.inf, 
                 max_time=np.inf, 
                 max_iterations=np.inf, 
                 default_policy = RandomPlayer(), 
                 name = "MCTS_Solver",
                  secure_child_A = 1,
                 logs = False):
        super().__init__(rollouts, c, max_fm, max_time, max_iterations, default_policy, name, logs)
        self.secure_child_A = secure_child_A
    
    def choose_action(self, state):

        self.choose_action_logs = pd.DataFrame()
        self.nodes_count = 0
        self.root_node = Node(state=state.duplicate(), expansion_index=self.nodes_count)
        self.nodes_count += 1
        self.player = self.root_node.state.player_turn
        self.current_fm = 0
        self.current_iterations = 0
        self.current_time = 0
        
        start_time = time.time()

        while self.current_fm < self.max_fm and self.current_iterations < self.max_iterations and self.current_time < self.max_time and self.root_node.average_reward() != np.inf:
            self.iteration(self.root_node)

            #Update criteria
            self.current_iterations = self.current_iterations + 1
            self.current_time = time.time() - start_time

            #Check if iterations are still calling fm
            if self.current_fm < self.current_iterations:
                print("Warning: current_fm >= current_iterations. Fm calls:", self.current_fm, "Its:",self.current_iterations)
                break

        if len(self.root_node.children) > 0:
            to_return = self.recommendation_policy()
        else:
            #print(self.name, ": Random move returned")
            to_return = rd.choice(self.root_node.state.available_actions)
        
        #Update logs
        if self.logs:
            self._update_choose_action_logs()
            self.choose_action_logs["chosen_action"] = to_return

        return to_return

    def expansion(self, node) -> Node:

        #Check if any of the immediately available states is terminal
        if len(node.children) == 0 and node.is_chance_node == False:
            for move in node.state.available_actions:
                state_duplicate = node.state.duplicate()
                state_duplicate.make_action(move)
                self.current_fm = self.current_fm + 1
                if state_duplicate.is_terminal:

                    #If the terminal state is a win for the player that makes the action, the node gets proven.
                    if state_duplicate.winner == node.state.player_turn:
                        expanded_node = node.add_child(edge_content=move, state=state_duplicate, expansion_index=self.nodes_count)
                        self.nodes_count += 1
                        return expanded_node

        return super().expansion(node)

    def simulation(self, node, rollouts, default_policy) -> float:

        #Return infinite rewards for win and loss if the expanded node was a terminal node
        if node.state.is_terminal:
            if node.state.winner == self.player:
                return np.inf
            elif node.state.winner is not None:
                return -np.inf

        return super().simulation(node, rollouts, default_policy)

    def backpropagation(self, node, reward) -> None:

        #reward to replace infinite rewards
        non_proven_reward = node.state.reward[self.player]

        while node.parent is not None:

            node.update(reward)
            node = node.parent

            #if the rewards are infinite, nodes will be proven loses/wins. 
            if abs(reward) == np.inf:

                #If the player can and wants to force a result, the node is proven and rewards are still infinite
                if (node.state.player_turn == self.player and reward == np.inf) or (node.state.player_turn != self.player and reward == -np.inf):
                    continue
                
                #If cannot force a desired result, check if the player has alternatives to escape and undesirable result
                #If there are alternatives, the node is not proven and rewards are set to non_proven_reward
                if node.can_be_expanded():
                    reward = non_proven_reward
                    continue
                
                #if reward is undesirable proven, check all the alternatives
                if node.state.player_turn == self.player and reward == -np.inf:
                    for child in node.children.values():
                        if child.average_reward() != -np.inf:
                            reward = non_proven_reward
                            break
                #same with inf
                elif node.state.player_turn != self.player and reward == np.inf:
                    for child in node.children.values():
                        if child.average_reward() != np.inf:
                            reward = non_proven_reward
                            break
            
        node.update(reward)

    def recommendation_policy(self, root_node=None):
        """Secure child recommendation policy"""
        if root_node is None:
            root_node = self.root_node
        children = root_node.children.values()
        rd.shuffle(children)
        return max(children, key= lambda x: x.average_reward() + self.secure_child_A/math.sqrt(x.visits)).edge_action

    def agent_data(self):
        agent_data = super().agent_data()
        data_dict = {
            "secure_child_A":self.secure_child_A
        }
        data_df = pd.DataFrame(data_dict, index=[0])
        return pd.concat([agent_data, data_df], axis=1)









