
from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import math
import numpy as np
import random as rd
import time
import Games.base_games as base_games

class Node():

    def __init__(self, state=None, parent = None):
        self.parent = parent
        self.state = state #Saving states accelerates iterations but increases memory usage

        self.visits = 0
        self.reward = 0
        self.children = {}
        
    def is_leaf(self):
        return len(self.children) == 0

    def UCB(self, c = math.sqrt(2)):

        if self.visits == 0:
            return np.inf
        else:
            return self.reward / self.visits + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def expand(self, action=None): #game interface dependencies

        duplicate_state = self.state.duplicate()
        if action is None:
            action = rd.choice(duplicate_state.available_actions)
            duplicate_state.make_action(action)
        else:
            duplicate_state.make_action(action) 

        new_node = Node(state = duplicate_state, parent = self)
        self.children[action] = new_node
        return new_node

    def update_visits(self):
        self.visits = self.visits + 1

    def update_reward(self, new_reward):
        self.reward = self.reward + new_reward

    def select_by_UCT(self, c):
        if self.is_leaf() or len(self.children) != len(self.state.available_actions):
            return self
        return max(self.children, key= lambda x: self.children[x].UCB(c))

    def select_by_reward(self):
        return max(self.children, key= lambda x: self.children[x].reward)

    def __str__(self):
        return f"visits {self.visits}, reward {self.reward}, children {len(self.children)}"

    def __eq__(self, other):
        return other.state == self.state

    def __hash__(self):
        return hash((self.state))

    def __ne__(self, other):
        return not(self == other)



class MCTS_Player(base_games.Agent):

    root_node : Node
    player : int = 0

    def __init__(self, rollouts, c, max_fm=np.inf, max_time=10, max_simulations=np.inf):

        self.rollouts = rollouts
        self.c = c
        self.max_fm = max_fm
        self.max_time = max_time
        self.max_simulations = max_simulations
        self.current_fm = 0
        self.current_simulations = 0
        self.current_time = 0

    def choose_action(self, state):

        self.root_node = Node(state=state)
        self.player = self.root_node.player_turn
        self.current_fm = 0
        self.current_simulations = 0
        self.current_time = 0
        start_time = time.time()
        agent = RandomPlayer()

        while self.current_fm < self.max_fm and self.current_simulations < self.max_simulations and self.current_time < self.max_time:

            node = root_node

            #Selection
            node = self.selection(node)

            #Expansion
            node = self.expansion(node)

            #Simulation
            reward = 0
            for _ in range(self.rollouts):
                reward = reward + self.map_reward(winner = self.simulation(agent, node))
                self.current_simulations = self.current_simulations + 1
            reward = reward / self.rollouts

            #Backpropagation
            self.backpropagation(node, reward)

            #Update criteria
            self.current_time = start_time - time.time()

        return root_node.select_by_reward()

    def map_reward(self, winner, win = 1, lose = -1, draw = 0):
        if winner == self.player:
            return win
        elif winner == base_games.next_player(self.player):
            return lose
        else:
            return draw

    def selection(self, node) -> Node:

        while not node.is_leaf():
            node = self.root_node.select_by_UCT(self.c)
        return node

    def expansion(self, node) -> Node:

        node = node.expand()
        self.current_fm = self.current_fm + 1
        return node

    def simulation(self, agent, node) -> int: #game interface dependencies

        state = node.state.duplicate()
        while not state.is_terminal():
            state = state.make_action(agent.choose_action(state))
            self.current_fm = self.current_fm + 1
        return state.winner()

    def backpropagation(self, node, reward) -> None:

        if node.state.player_turn == self.player:
            node.update_reward(reward)
        else:
            node.update_reward(-reward)
        node.update_visits()

        if node.parent is not None:
            self.backpropagation(node.parent, reward)
            



