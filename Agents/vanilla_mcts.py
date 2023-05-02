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
        self.children = {} #action:node

    def can_be_expanded(self):
        return len(self.state.available_actions) > len(self.children) and len(self.state.available_actions) > 0 and not self.state.is_terminal

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
        return "edge_action:" + str(self.edge_action) + ", visits:" + str(self.visits) + ", avg_reward:" + "{0:.3g}".format(self.total_reward/self.visits) + ", children:" + str(len(self.children))

    def __eq__(self, other):
        return other.state == self.state

    def __hash__(self):
        return hash((self.state))

    def __ne__(self, other):
        return not(self == other)



class MCTS_Player(BaseAgent):

    root_node : Node
    player : int = 0

    def __init__(self, rollouts=1, c=math.sqrt(2), max_fm=np.inf, max_time=np.inf, max_iterations=np.inf, default_policy = RandomPlayer(), name = "Vanilla_MCTS", logs = False):
        assert max_fm != np.inf or max_time != np.inf or max_iterations != np.inf, "At least one of the stopping criteria must be set"
        self.rollouts = rollouts
        self.c = c
        self.max_fm = max_fm
        self.max_time = max_time
        self.max_iterations = max_iterations
        self.current_fm = 0
        self.current_iterations = 0
        self.current_time = 0
        self.nodes_count = 0
        self.default_policy = default_policy
        self.name = name
        self.logs = logs

    def choose_action(self, state):

        self.root_node = Node(state=state.duplicate())
        self.player = self.root_node.state.player_turn
        self.current_fm = 0
        self.current_iterations = 0
        self.current_time = 0
        self.nodes_count = 0
        start_time = time.time()

        while self.current_fm < self.max_fm and self.current_iterations < self.max_iterations and self.current_time < self.max_time:
            self.iteration(self.root_node)

            #Update criteria
            self.current_iterations = self.current_iterations + 1
            self.current_time = start_time - time.time()

        if len(self.root_node.children) > 0:
            return max(self.root_node.children.values(), key= lambda x: x.visits).edge_action
        else:
            print(self.name, ": Random move returned")
            return rd.choice(self.root_node.state.available_actions)

    def iteration(self, node):

        #Selection
        node = self.selection(node)

        #Expansion
        if node.can_be_expanded():
            node = self.expansion(node)

        #Simulation
        reward = self.simulation(node, self.rollouts, self.default_policy) 

        #Backpropagation
        self.backpropagation(node, reward)

    def selection(self, node) -> Node:
        #Returns a node that can be expanded selecting by UCT
        assert node.state.is_terminal == False, "Selection called on a terminal node"
        while not node.can_be_expanded() and not node.state.is_terminal: 
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

    def simulation(self, node, rollouts, default_policy) -> float:
        #Returns the average reward of the rollouts

        #State was terminal
        if node.state.is_terminal:
            return state.score[self.player] #change to score

        #Execute simulations
        reward = 0
        for _ in range(rollouts):
            state = node.state.duplicate()
            while not state.is_terminal:
                state.make_action(default_policy.choose_action(state))
                self.current_fm = self.current_fm + 1
            reward = reward + state.score[self.player]
        average_reward = reward / self.rollouts
        return average_reward

    def backpropagation(self, node, reward) -> Node:

        while node.parent is not None:
            node.update(reward)
            node = node.parent
        node.update(reward)
        
    def UCB1(self, node, c = math.sqrt(2)):
        if node.visits == 0:
            return np.inf
        else:
            assert node.parent is not None

            #Verify who's turn it is
            reward = node.total_reward if node.parent.state.player_turn == self.player else -node.total_reward

            #UCB1
            return reward / node.visits + c * math.sqrt(math.log(node.parent.visits) / node.visits)
            
    def view_mcts_tree(self, node=None, depth=0):
        if node is None:
            node = self.root_node
        if depth != 0:
            ucb = "{0:.3g}".format(self.UCB1(node))
        else: ucb = "None"

        my_string = f"\n{'--'*depth}{str(node)} ucb:" + str(ucb)

        for child in node.children.values():
            my_string = my_string + self.view_mcts_tree(child, depth+1)
        return my_string

    def __str__(self):
        return self.name


