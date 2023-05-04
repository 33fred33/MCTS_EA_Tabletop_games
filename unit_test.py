import cProfile
import random as rd
import time
from collections import defaultdict
import numpy as np

#import Games.othello as oth
import Games.mnk as mnk
import Games.function_optimisation as fo
import Agents.random as arand
import Agents.vanilla_mcts as mcts
import Agents.siea_mcts as siea_mcts
import Utilities.experiment_utils as eu

def game_can_end(state, random_seed, max_turns = 10000):
    for t in range(max_turns):
        if state.is_terminal:
            assert state.turn > 1, "Game ended in first turn. Random seed: " + str(random_seed) + ". Implies duplication error"
            return True
        state.make_action(rd.choice(state.available_actions))
    return False

def game_can_end_with_agents(state, agents, random_seed, max_turns = 10000):
    for t in range(max_turns):
        if state.is_terminal:
            agent_names = [str(agent) for agent in agents]
            assert state.turn > 1, "Game ended in first turn. Random seed: " + str(random_seed) + ". Agents" + str(agent_names)
            return True
        state.make_action(agents[state.player_turn].choose_action(state))
    return False

def test_game(state):
    #Run full games
    print("Testing game" + str(state))
    for rs in range(100):
        rd.seed(rs)
        duplicate_state = state.duplicate()
        assert game_can_end(duplicate_state, rs), "Game did not end in 10000 turns. Random seed: " + str(rs)
    return True

def test_agent(agent):
    state = produce_game("mnk")
    random_agent = produce_agent("random")
    print("Testing agent" + str(agent) + " as second player")
    for rs in range(50):
        rd.seed(rs)
        duplicate_state = state.duplicate()
        assert game_can_end_with_agents(duplicate_state, [random_agent, agent], rs), "Game did not end in 10000 turns. Random seed: " + str(rs) + ". Agent: " + str(agent) + "as second player"
    print("Testing agent" + str(agent) + " as first player")
    for rs in range(50):
        rd.seed(rs)
        duplicate_state = state.duplicate()
        assert game_can_end_with_agents(duplicate_state, [agent, random_agent], rs), "Game did not end in 10000 turns. Random seed: " + str(rs) + ". Agent: " + str(agent) + "as first player"
    return True

def produce_agent(name):
    if name=="random": return arand.RandomPlayer()
    elif name=="mcts": return mcts.MCTS_Player(max_iterations=100, logs=True)
    elif name=="siea_mcts": return siea_mcts.SIEA_MCTS_Player(max_iterations=100, logs=True)
    else: print("Agent name not recognised")

def produce_game(name):
    if name=="mnk": state = mnk.GameState(m=3,n=3,k=3)
    elif name=="gomoku": state = mnk.GameState(m=13,n=13,k=5)
    elif name=="fo1d1p": state = fo.GameState(function_index=0, n_players=1)
    elif name=="fo1d2p": state = fo.GameState(function_index=0, n_players=2)
    elif name=="fo2d1p": state = fo.GameState(function_index=5, n_players=1)
    elif name=="fo2d2p": state = fo.GameState(function_index=5, n_players=2)
    else: print("Game name not recognised")
    state.set_initial_state()
    return state

def test_mcts_tree(iterations=20, game_name = "mnk", agent_name = "mcts"):
    state = produce_game(game_name)
    agent = produce_agent(agent_name)
    agent.max_iterations = iterations#-1
    #_ = agent.choose_action(state)
    #for _ in range(iterations):
    #    agent.iteration(agent.root_node)
    _ = agent.choose_action(state)
    print(agent.view_mcts_tree())
    return agent

def run():
    #Database
    game_names = ["mnk", "fo1d1p", "fo1d2p", "fo2d1p", "fo2d2p"]
    agent_names = ["random", "mcts", "siea_mcts"]

    #Test game mnk
    print("Game tests running")
    for game in [produce_game(name) for name in game_names]:
        assert test_game(game)
    print("Game tests passed")

    #Test agents
    print("Agent tests running")
    for agent in [produce_agent(name) for name in agent_names]:
        assert test_agent(agent)
    print("Agent tests passed")

    #Test experiment_utils
    print("Experiment utils tests running")
    state = produce_game("mnk")
    test_game_player = eu.GamePlayer(state, [produce_agent(agent_names[0]),produce_agent(agent_names[0])])
    test_game_player.play_game()
    test_game_player.play_games(n_games=4)
    print("Play_game logs:")
    print(test_game_player.logs_by_game)
    print("Experiment utils tests passed")


   