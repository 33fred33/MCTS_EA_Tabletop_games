import cProfile
import random as rd
import time
from collections import defaultdict
import numpy as np

#import Games.othello as oth
import Games.mnk as mnk
import Agents.random as arand
import Agents.vanilla_mcts as mcts
import Utilities.experiment_utils as eu

def game_can_end(state, random_seed, max_turns = 10000):
    for t in range(max_turns):
        if state.is_terminal:
            assert state.turn > 1, "Game ended in first turn. Random seed: " + str(random_seed) + ". Implies duplication error"
            return True
        state.make_action(rd.choice(state.available_actions))
    return False

def test_game(state):
    state.set_initial_state()
    for rs in range(100):
        rd.seed(rs)
        duplicate_state = state.duplicate()
        assert game_can_end(duplicate_state, rs), "Game did not end in 10000 turns. Random seed: " + str(rs)
    return True

def produce_agent(name):
    if name=="random": return arand.RandomPlayer()
    if name=="mcts": return mcts.VanillaMCTSPlayer()

def produce_game(name):
    if name=="mnk": return mnk.GameState(m=3,n=3,k=3)
    if name=="gomoku": return mnk.GameState(m=13,n=13,k=5)

def unit_tests():

    game_names = ["mnk"]
    agent_names = ["random", "mcts"]

    #Test game mnk
    for game in [produce_game(name) for name in game_names]:
        assert test_game(game)

    #Test experiment_utils
    assert eu.play_game(produce_game[game_names[0]], [produce_agent(agent_names[0]),produce_agent(agent_names[0])])[0].is_terminal, "Game did not end with random agents"

    #Test agents
    for agent in [produce_agent(name) for name in agent_names]:
        assert eu.play_game(produce_game[game_names[0]], [agent, produce_agent(agent_names[0])])[0].is_terminal, "Game did not end with agent" + str(agent)

   