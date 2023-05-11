import random
import itertools as it
from scipy.stats import bernoulli
import numpy as np
import math
from typing import List
import pandas as pd

import Games.base_games as base_games

def f0(x):
    """Unimodal, centered"""
    return math.sin(math.pi*x[0])
def f1(x):
    """Multimodal, paper bubeck"""
    return 1/2*(math.sin(13*x[0])*math.sin(27*x[0])+1)
def f2(x):
    """Smoothness with levels, paper finnsson"""
    if x[0] < 0.5:
        if x[0] > 0:
            return 0.5+0.5*abs(math.sin(1/pow(x[0],5)))
        else:
            return 0.5+0.5*abs(math.sin(1/0.000001))
    else:
        return 7/20+0.5*abs(math.sin(1/pow(x[0],5)))
def f3(x):
    """Deceptive"""
    return (0.5*x[0])+(-0.7*x[0]+1)*pow(math.sin(5*math.pi*x[0]),4)
def f4(x):
    """Deceptive, search traps"""
    return (0.5*x[0])+(-0.7*x[0]+1)*pow(math.sin(5*math.pi*x[0]),80)
def f5(x):
    """Two variables: smooth"""
    return f0([x[0]])*f0([x[1]])
def f6(x):
    """Deceptive, search traps"""
    return f1([x[0]])*f1([x[1]])
def f7(x):
    """Deceptive, search traps"""
    return f2([x[0]])*f2([x[1]])
def f8(x):
    """Deceptive, search traps"""
    return f3([x[0]])*f3([x[1]])
def f9(x):
    """Deceptive, search traps"""
    return f4([x[0]])*f4([x[1]])
def f10(x):
    """Deceptive, search traps"""
    return f3([x[0]])*f1([x[1]])
function_list=[f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
dimensions = [1,1,1,1,1,2,2,2,2,2]
db_ranges = [[[0,1]] for _ in range(5)] + [[[0,1],[0,1]] for _ in range(5)]
max_location_list = [[0.5],[0.86832],[None],[0.1],[0.1],[0.5,0.5],[0.866,0.866],[None,None],[0.1,0.1],[0.1,0.1]]

class Action():

    def __init__(self, player_index, ranges):
        self.player_index = player_index
        self.ranges = ranges

    def __str__(self):
        return str([["{:.3f}".format(x) for x in d] for d in self.ranges]) + "t" + str(self.player_index)
    
    def __members(self):
        l = []
        for d in self.ranges:
           l = l+ [d[0],d[1]]
        return tuple(x for x in l) + (self.player_index,)

    def __eq__(self, other):
        return self.__members() == other.__members()

    def __hash__(self):
        return hash(self.__members())


class GameState(base_games.BaseGameState):
    """
    List of important attributes:
        self.players: list of player objects (min len: 1, max len: 2)
        self.winner: Result at end of game. 1=P1 wins, 2=P2 wins, 0=Draw
        self.reward: state's value
        self.player_turn: Indicates the current player
        self.is_terminal: Indicates if the state is terminal
    Assumes ranges are 0 to 1 for any dimension, and reward goes from 0 to 1
    First player maximises, second player minimises
    """

    def __init__(self, n_players=1, function_index=0, splits=2, minimum_step=0.00001, max_turns=np.inf, for_test=False, name = None):
        """
        players: list of player objects (min len: 1, max len: 2)
        function: fitness method (takes a list of len="dimensions" as argument). If is an int, default functions will be used
        ranges: list (dimensions; min len: 1, max len: any) of lists (min and max; len: 2) of domains.
        splits: (int) equal split amount
        """
        assert n_players in [1,2], "n_players must be 1 or 2"
        assert function_index in range(len(function_list)), "function_index must be in range(len(function_list))"
        assert splits >= 2, "splits must be >= 2"

        # assignation
        self.n_players = n_players
        self.function_index = function_index
        self.dimensions = dimensions[function_index]
        self.initial_ranges = db_ranges[function_index]
        self.splits = splits
        self.minimum_step = minimum_step
        self.max_turns = max_turns
        self.for_test = for_test
        self.function = function_list[function_index]
        self.max_location = max_location_list[function_index]
        if name is None:
            self.name = "FOP_f" + str(function_index) + "_s" + str(splits) + "_p" + str(n_players)
        else: self.name = name
         
    def set_initial_state(self) -> None:
        self.winner = None
        self.reward = [None for _ in range(self.n_players)]
        self.player_turn = 0
        self.is_terminal = False
        self.turn = 1
        self.ranges = self.initial_ranges
        self._update_available_actions()
    
    def duplicate(self):
        """
        Clones the game state - quicker than using copy.deepcopy()
        """
        Clone = GameState(n_players=self.n_players, 
                        function_index = self.function_index,
                        splits = self.splits, 
                        minimum_step = self.minimum_step, 
                        max_turns = self.max_turns, 
                        for_test = self.for_test,
                        name = self.name)
        Clone.available_actions = [a for a in self.available_actions]
        Clone.ranges = [[x for x in d] for d in self.ranges]
        Clone.winner = self.winner
        Clone.reward = [s for s in self.reward]
        Clone.player_turn = self.player_turn
        Clone.is_terminal = self.is_terminal   
        Clone.turn = self.turn
        return Clone
   
    def eval_point(self):
      return [(r[0] + r[1])/2 for r in self.ranges]

    def make_action(self, action = None):
      """
      Place a move on the game board
         Move: list (dimensions; min len: 1, max len: any) of lists (min and max; len: 2) of domains.
      """

      #Update state
      self.ranges = [[x for x in d] for d in action.ranges]

      #Verify termination
      if self.ranges[0][1] - self.ranges[0][0] < self.minimum_step or self.turn > self.max_turns:
         self.is_terminal = True
         p = self.function(self.eval_point())
         if self.for_test: 
             if self.n_players == 1: self.reward[0] = p #for finding the actual max
             else:
                    self.reward[0] = p
                    self.reward[1] = 1 - p
         else: 
             if self.n_players == 1: self.reward[0] = bernoulli.rvs(p) #for finding the actual max
             else:
                    evaluation = bernoulli.rvs(p)
                    self.reward[0] = evaluation
                    self.reward[1] = 1 - evaluation
         return
      
      #turn end routine
      if self.n_players == 2:
         self.player_turn = 1 - self.player_turn # switch turn
      self.turn = self.turn+ 1  # increment turns
      self._update_available_actions()

    def _update_available_actions(self):
        if self.ranges[0][1] - self.ranges[0][0] < self.minimum_step or self.turn > self.max_turns:
            return []

        available_ranges = []
        for r in self.ranges:
            dimension_ranges = []
            for s in range(self.splits):
                start = r[0] + (r[1]-r[0])*s/self.splits
                finish = r[0] + (r[1]-r[0])*(s+1)/self.splits
                dimension_ranges.append([start, finish])
            available_ranges.append(dimension_ranges)
        available_combinations = list(it.product(*available_ranges))
        self.available_actions = [None for _ in range(len(available_combinations))]
        for i,ac in enumerate(available_combinations):
            self.available_actions[i] = Action(self.player_turn, [[x for x in d] for d in ac])
        #print([str(a) for a in self.available_actions])

    def feature_vector(self):
        """Features: player turn, ranges0, ranges1"""
        features = {"Player_symbol":self.player_turn}
        for d in range(len(self.ranges)):
            features["Dimension"+str(d) + "_start"] = self.ranges[d][0]
            features["Dimension"+str(d) + "_end"] = self.ranges[d][1]
        return features
    
    def game_definition_data(self):
        data = {
            "Name": self.name,
            "Function_index": self.function_index,
            "Splits": self.splits,
            "Minimum_step": self.minimum_step,
            "Max_turns": self.max_turns,
            "For_test": self.for_test,
            "N_players": self.n_players,
            "Max_location": str(self.max_location),
            "Dimensions": str(self.dimensions),
        }
        return pd.DataFrame(data, index=[0])

    def logs_data(self):
        data = self.feature_vector()
        for i, player_reward in enumerate(self.reward):
            data["Score_p"+str(i)] = player_reward
        for d in range(len(self.ranges)):
            data["Eval_point_dimension"+str(d)] = self.eval_point()[d]
        data["Turn"] = self.turn
        data["Winner"] = self.winner
        data["Eval_point"] = str(self.eval_point())
        data["Is_terminal"] = self.is_terminal
        data["N_available_actions"] = len(self.available_actions)
        return pd.DataFrame(data, index=[0])

    def __repr__(self):
      return self.name + str(self.feature_vector())

             
            