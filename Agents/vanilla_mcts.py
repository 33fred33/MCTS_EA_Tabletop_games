from Agents.base_agents import BaseAgent
from Agents.random import RandomPlayer
from dataclasses import dataclass
from enum import Enum, auto
from typing import List
import math
import numpy as np
import random as rd
import time
import pandas as pd
import Utilities.logs_management as lm

class Node():

    def __init__(self, state=None, parent = None, edge_action = None, expansion_index = None, is_chance_node = False):
        self.parent = parent
        self.state = state #Saving states accelerates iterations but increases memory usage
        self.expansion_index = expansion_index #For logs
        self.edge_action = edge_action
        self.is_chance_node = is_chance_node

        self.visits = 0
        self.total_reward = 0
        self.children = {} #action:node

    def duplicate(self, parent = None):
        #Duplicates node and subtree
        clone_node = Node(state=self.state.duplicate(), parent=parent, edge_action=self.edge_action, expansion_index=self.expansion_index, is_chance_node=self.is_chance_node)
        clone_node.visits = self.visits
        clone_node.total_reward = self.total_reward
        for action, child in self.children.items():
            clone_child_node = child.duplicate(parent=clone_node)
            clone_node.children[action] = clone_child_node
        return clone_node

    def can_be_expanded(self):
        return len(self.state.available_actions) > len(self.children) and len(self.state.available_actions) > 0 and not self.state.is_terminal

    def average_reward(self):
        return self.total_reward/self.visits if self.visits > 0 else np.nan

    def random_available_action(self):
        assert self.can_be_expanded(), "Node cannot be expanded"
        return rd.choice([a for a in self.state.available_actions if a not in self.children])

    def add_child(self, edge_content, state, expansion_index, is_chance_node=False):
        """Adds a child to the node, and returns the child."""
        child_node = Node(state=state, parent=self, edge_action = edge_content, expansion_index=expansion_index, is_chance_node=is_chance_node)
        self.children[edge_content] = child_node
        return child_node

    def sample_random_event(self, expansion_index):
        """Samples the random event from the edge, and adds a child if the event was not already a child. Returns the chosen node. Equivalent to expand a decision node"""
        assert self.is_chance_node, "Sample random event called on a decision node"
        dupe_state = self.state.duplicate()
        random_event = dupe_state.sample_random_event(self.edge_action.event_type)
        for child in self.children:
            if child.edge_action == random_event:
                return child, False
        new_node = self.add_child(edge_content=random_event, state=dupe_state, expansion_index=expansion_index, is_chance_node=True)
        return new_node, True
        
    def update(self, new_reward):
        self.total_reward = self.total_reward + new_reward
        self.visits = self.visits + 1

    def subtree_nodes(self, node=None):
        if node is None:
            node = self
        my_nodes = [node]

        for child in node.children.values():
            my_nodes = my_nodes + self.subtree_nodes(child)
        return my_nodes

    def node_data(self):
        data = {
            "expansion_index": self.expansion_index,
            "visits": self.visits,
            "total_reward": self.total_reward,
            "avg_reward": self.total_reward/self.visits if self.visits > 0 else np.nan,
            "children": len(self.children),
            "state": str(self.state),
            "edge_action": str(self.edge_action),
            "is_chance_node": str(self.is_chance_node),
        }
        return pd.DataFrame(data, index=[0])

    def __str__(self):
        avg_reward = self.total_reward/self.visits if self.visits > 0 else np.nan
        return "edge_action:" + str(self.edge_action) + ", visits:" + str(self.visits) + ", avg_reward:" + "{0:.3g}".format(avg_reward) + ", children:" + str(len(self.children)) + ", is_chance_node:" + str(self.is_chance_node) + ", expansion_index:" + str(self.expansion_index)

    def __eq__(self, other):
        if other is None:
            #print("Warning: comparing node to None")
            return False
        return other.state == self.state

    def __hash__(self):
        return hash((self.state))

    def __ne__(self, other):
        return not(self == other)


class MCTS_Player(BaseAgent):

    root_node : Node
    player : int

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
        self.choose_action_logs = pd.DataFrame()
        self.isAIPlayer = True

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

        while self.current_fm < self.max_fm and self.current_iterations < self.max_iterations and self.current_time < self.max_time:
            self.iteration(self.root_node)

            #Update criteria
            self.current_iterations = self.current_iterations + 1
            self.current_time = time.time() - start_time

            #Check if iterations are still calling fm
            if self.current_fm < self.current_iterations:
                print("Warning: current_fm >= current_iterations. Fm calls:", self.current_fm, "Its:",self.current_iterations)
                break

        if len(self.root_node.children) > 0:
            to_return = max(self.root_node.children.values(), key= lambda x: x.visits).edge_action
        else:
            print(self.name, ": Random move returned")
            to_return = rd.choice(self.root_node.state.available_actions)
        
        #Update logs
        if self.logs:
            self._update_choose_action_logs()
            self.choose_action_logs["chosen_action"] = to_return

        return to_return

    def iteration(self, node):

        #Selection
        node = self.selection(node) #this will always return a decision node

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
        assert node.is_chance_node == False, "Selection called on a chance node"
        while not node.can_be_expanded() and not node.state.is_terminal:

            node = self.best_child_by_tree_policy(node.children.values())
            self.current_fm = self.current_fm + 1

            #if node is chance node, return a random child
            while node.is_chance_node:
                node, added_chance_node = node.sample_random_event(expansion_index = self.nodes_count)
                if added_chance_node:
                    self.nodes_count += 1

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

        #Add chance nodes. Assumes that random events have already occurred in the game state, and adds that event as an additional node
        if hasattr(duplicate_state, "random_events"):
            random_events_copy = [e.duplicate() for e in duplicate_state.random_events]
            while len(random_events_copy) > 0:
                chance_duplicate_state = duplicate_state.duplicate()
                #Add chance node
                new_node = new_node.add_child(edge_content = random_events_copy.pop(0), state = chance_duplicate_state, expansion_index=self.nodes_count, is_chance_node=True)
                self.nodes_count += 1

        return new_node

    def simulation(self, node, rollouts, default_policy) -> float:
        #Returns the average reward of the rollouts

        #State was terminal
        if node.state.is_terminal:
            return node.state.reward[self.player]

        #Execute simulations
        reward = 0
        for _ in range(rollouts):
            state = node.state.duplicate()
            while not state.is_terminal:
                state.make_action(default_policy.choose_action(state))
                self.current_fm = self.current_fm + 1
            reward = reward + state.reward[self.player]
        average_reward = reward / self.rollouts
        return average_reward

    def backpropagation(self, node, reward) -> Node:

        while node.parent is not None:
            node.update(reward)
            node = node.parent
        node.update(reward)
        
    def tree_policy_formula(self, node):
        if node.visits == 0:
            return np.inf
        else:
            assert node.parent is not None

            #Verify who's turn it is
            reward = node.total_reward if node.parent.state.player_turn == self.player else -node.total_reward

            #UCB1
            return reward / node.visits + self.c * math.sqrt(math.log(node.parent.visits) / node.visits)

    def best_child_by_tree_policy(self, children, my_tree_policy_formula=None):
        """
        Children is a list of nodes
        Returns a node with the maximum tree policy value. If there are more than one, returns a random one
        """
        assert len(children) > 0, "No children to choose from"
        if my_tree_policy_formula is None: 
            my_tree_policy_formula = self.tree_policy_formula
        best_value = -np.inf
        best_children = []
        for child in children:
            value = my_tree_policy_formula(child)
            if value > best_value:
                best_value = value
                best_children = [child]
            elif value == best_value:
                best_children.append(child)
        assert len(best_children) > 0, "No best children found"
        return rd.choice(best_children)
        
    def view_mcts_tree(self, node=None, depth=0):
        if node is None:
            node = self.root_node
        if depth != 0:
            ucb = "{0:.3g}".format(self.tree_policy_formula(node))
        else: ucb = "None"

        my_string = f"\n{'--'*depth}{str(node)}, tree_policy_formula:" + str(ucb)

        for child in node.children.values():
            my_string = my_string + self.view_mcts_tree(child, depth+1)
        return my_string

    def agent_data (self):
        data_dict = {
            "Player": str(self),
            "player_name": self.name,
            "c_parameter": self.c,
            "rollouts": self.rollouts,
            "max_fm": self.max_fm,
            "max_time": self.max_time,
            "max_iterations": self.max_iterations,
            "default_policy": str(self.default_policy),
            "producing_logs": self.logs,
        }
        return pd.DataFrame(data_dict, index=[0])

    def _update_choose_action_logs(self):
        data_dict = {
            "state": str(self.root_node.state),
            "forward_model_calls": self.current_fm,
            "current_time": self.current_time,
            "current_iterations": self.current_iterations,
            "root_node_visits": str(self.root_node.visits),
            "root_node_avg_reward": self.root_node.total_reward / self.root_node.visits if self.root_node.visits > 0 else np.nan ,
            "nodes_count": self.nodes_count,
            #"root_nodes_count": self.root_node.subtree_nodes(), 
        }
        for i,c in enumerate(self.root_node.children.values()):
            data_dict["action" + str(i)] = str(c.edge_action)
            data_dict["action" + str(i) + "_is_chance_node"] = c.is_chance_node
            data_dict["action" + str(i) + "_visits"] = c.visits
            data_dict["action" + str(i) + "_avg_reward"] = c.total_reward / c.visits if c.visits > 0 else np.nan   
            data_dict["action" + str(i) + "_tree_policy_formula"] = self.tree_policy_formula(c)
        action_df = pd.DataFrame(data_dict, index=[0])
        action_df = pd.concat([action_df, self.agent_data()], axis=1)
        self.choose_action_logs = pd.concat([self.choose_action_logs, action_df], axis=1)

    def dump_my_logs(self, path):
        lm.dump_data(self.agent_data(), path, "agent_data.csv")
        lm.dump_data(self.choose_action_logs, path, "choose_action_logs.csv")

    def __str__(self):
        return self.name


