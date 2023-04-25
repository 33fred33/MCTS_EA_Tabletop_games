from Agents.base_agents import BaseAgent
from Agents.random import RandomPlayer
from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import math
import numpy as np
import random as rd
import time

class Node():

    def __init__(self, state=None, parent = None, edge_action = None, expansion_index = None):
        self.parent = parent
        self.state = state #Saving states accelerates iterations but increases memory usage
        self.expansion_index = expansion_index #For logs
        self.edge_action = edge_action

        self.visits = 0
        self.total_reward = 0
        self.children = {}
        
    def is_leaf(self):
        return self.state.is_terminal

    def can_be_expanded(self):
        return len(self.state.available_actions) > len(self.children)

    def random_available_action(self):
        return rd.choice([a for a in self.state.available_actions if a not in self.children])

    def add_child(self, action, state, expansion_index):
        child_node = Node(state=state, parent=self, edge_action = action, expansion_index=expansion_index)
        self.children[action] = child_node
        return child_node

    def update(self, new_reward):
        self.total_reward = self.total_reward + new_reward
        self.visits = self.visits + 1

    def __str__(self):
        return f"visits {self.visits}, reward {self.total_reward}, children {len(self.children)}"

    def __eq__(self, other):
        return other.state == self.state

    def __hash__(self):
        return hash((self.state))

    def __ne__(self, other):
        return not(self == other)



class MCTS_Player(BaseAgent):

    root_node : Node
    player : int = 0

    def __init__(self, rollouts, c, max_fm=np.inf, max_time=np.inf, max_iterations=np.inf, default_policy = RandomPlayer()):
        assert max_fm != np.inf or max_time != np.inf or max_iterations != np.inf, "At least one of the stopping criteria must be set"
        self.rollouts = rollouts
        self.c = c
        self.max_fm = max_fm
        self.max_time = max_time
        self.max_simulations = max_iterations
        self.current_fm = 0
        self.current_iterations = 0
        self.current_time = 0
        self.default_policy = default_policy

    def choose_action(self, state):

        self.root_node = Node(state=state)
        self.player = self.root_node.state.player_turn
        self.current_fm = 0
        self.current_iterations = 0
        self.current_time = 0
        self.nodes_count = 0
        start_time = time.time()

        while self.current_fm < self.max_fm and self.current_iterations < self.max_simulations and self.current_time < self.max_time:

            node = self.root_node

            #Selection
            node = self.selection(node)

            #Expansion
            node = self.expansion(node)

            #Simulation
            reward = self.simulation(node, self.rollouts, self.default_policy) 

            #Backpropagation
            self.backpropagation(node, reward)

            #Restart
            node = self.root_node

            #Update criteria
            self.current_iterations = self.current_iterations + 1
            self.current_time = start_time - time.time()

        return max(self.root_node.children.values(), key= lambda x: x.visits).edge_action

    def selection(self, node) -> Node:
        #Returns a node that can be expanded selecting by UCT
        while not node.can_be_expanded():
            node = max(node.children.values(), key= lambda x: self.UCB1(x))
        return node

    def expansion(self, node) -> Node:
        #Returns a new node with a random action

        #Select random action
        action = node.random_available_action()
        
        #New state
        duplicate_state = node.state.duplicate()
        duplicate_state.make_action(action)

        #Add node to tree
        new_node = node.add_child(action, duplicate_state, expansion_index=self.nodes_count)
        self.nodes_count += 1
        self.current_fm = self.current_fm + 1

        return new_node

    def simulation(self, node, rollouts, default_policy) -> int:
        reward = 0
        for _ in range(rollouts):
            state = node.state.duplicate()
            while not state.is_terminal:
                state.make_action(default_policy.choose_action(state))
                self.current_fm = self.current_fm + 1
            reward = reward + self.map_reward(state.winner)
        average_reward = reward / self.rollouts
        return average_reward

    def backpropagation(self, node, reward) -> Node:

        while node.parent is not None:
            node.update(reward)
            node = node.parent
        node.update(reward)
    
    def map_reward(self, winner, win = 1, lose = -1, draw = 0):
        if winner == self.player:
            return win
        elif winner is None:
            return draw
        else:
            return lose
        
    def UCB1(self, node, c = math.sqrt(2)):
        if node.visits == 0:
            return np.inf
        else:
            assert node.parent is not None

            #Verify who's turn it is
            reward = node.total_reward if node.parent.state.player_turn == self.player else -node.total_reward

            #UCB1
            return reward / node.visits + c * math.sqrt(math.log(node.parent.visits) / node.visits)
            




