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


class MCTS_RAVE(MCTS_Player):

    def __init__(self, 
                 rollouts=1, 
                 c=math.sqrt(2), 
                 max_fm=np.inf, 
                 max_time=np.inf, 
                 max_iterations=np.inf, 
                 default_policy = RandomPlayer(), 
                 name = "MCTS_RAVE",
                    rave_bias = 0.5,
                 logs = False):
        super().__init__(rollouts, c, max_fm, max_time, max_iterations, default_policy, name, logs)
        self.rave_bias = rave_bias

    def simulation(self, node, rollouts, default_policy) -> float:
        #Returns the average reward of the rollouts. Updates amaf rewards of nodes

        #Collect nodes that could have their AMAF updated
        temp_node = node
        amaf_nodes = [] #nodes that could have their AMAF updated in this simulation
        must_update_amaf_nodes = [] #nodes that must have their AMAF updated in this simulation

        #Coming from node to root
        while temp_node.parent is not None:
            #Only care about nodes whose parent is not chance nodes
            if not temp_node.parent.is_chance_node:
                for n in temp_node.parent.children.values():
                    #If the node is not the one we are coming from
                    if n.edge_action != temp_node.edge_action:
                        #If the node's action is already in the AMAF list, we must update it
                        if n.edge_action in [e.edge_action for e in amaf_nodes]:
                            must_update_amaf_nodes.append(n)
                        #If the node's action is not in the AMAF list, we will keep it in the list of nodes to check
                        else:
                            amaf_nodes.append(n)
            temp_node = temp_node.parent

        #State was terminal
        if node.state.is_terminal:
            return node.state.reward[self.player]

        #Execute simulations
        reward = 0
        for _ in range(rollouts):
            state = node.state.duplicate()
            action_list = []
            while not state.is_terminal:
                action = default_policy.choose_action(state)
                action_list.append(action)
                state.make_action(action)
                self.current_fm = self.current_fm + 1
            
            #Get reward
            rollout_reward = state.reward[self.player]

            #Update RAVE moves
            for a in action_list:
                for n in amaf_nodes:
                    if n.edge_action == a:
                        n.update_amaf(rollout_reward)
            for n in must_update_amaf_nodes:
                n.update_amaf(rollout_reward)

            #Update reward
            reward = reward + rollout_reward

        average_reward = reward / self.rollouts
        return average_reward
    
    def tree_policy_formula(self, node): #UPDATING THIS -> FOLLOW FORMULAS FROM PAPER
        if node.visits == 0:
            return np.inf
        else:
            assert node.parent is not None

            b = node.amaf_visits/(node.visits + node.amaf_visits + 4*node.visits*node.amaf_visits*(self.rave_bias**2))

            #Verify who's turn it is
            reward = node.average_reward() if node.parent.state.player_turn == self.player else -node.average_reward()
            amaf_reward = node.average_amaf_reward() if node.parent.state.player_turn == self.player else -node.average_amaf_reward()

            #UCT_RAVE
            return (1-b)*reward + b*amaf_reward + self.c * math.sqrt(math.log(node.parent.visits) / node.visits)

    def agent_data(self):
        agent_data = super().agent_data()
        data_dict = {
            "rave_bias":self.rave_bias
        }
        data_df = pd.DataFrame(data_dict, index=[0])
        return pd.concat([agent_data, data_df], axis=1)









