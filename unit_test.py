import cProfile
import random as rd
import time
from collections import defaultdict
import numpy as np
import os

#import Games.othello as oth
import Games.mnk as mnk
import Games.function_optimisation as fo
import Agents.random as arand
import Agents.vanilla_mcts as mcts
import Agents.siea_mcts as siea_mcts
import Utilities.experiment_utils as eu
import Utilities.logs_management as lm

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
        chosen_action = agents[state.player_turn].choose_action(state)
        state.make_action(chosen_action)
        #print("Turn", str(t))
    return False

def test_game(state):
    #Run full games
    print("Testing game" + str(state))

    #Test duplicate not paired
    gs_dupe = state.duplicate()
    gs_dupe.make_action(rd.choice(gs_dupe.available_actions))
    assert state.turn != gs_dupe.turn

    #Test game can end
    for rs in range(100):
        rd.seed(rs)
        duplicate_state = state.duplicate()
        assert game_can_end(duplicate_state, rs), "Game did not end in 10000 turns. Random seed: " + str(rs)
    return True

def test_agent(agent):
    state = produce_game("mnk")
    random_agent = produce_agent("random")
    print("Testing agent" + str(agent) + " as second player")
    for rs in range(50,100):
        rd.seed(rs)
        duplicate_state = state.duplicate()
        assert game_can_end_with_agents(duplicate_state, [random_agent, agent], rs), "Game did not end in 10000 turns. Random seed: " + str(rs) + ". Agent: " + str(agent) + "as second player"
    print("Testing agent" + str(agent) + " as first player")
    for rs in range(50,100):
        rd.seed(rs)
        duplicate_state = state.duplicate()
        assert game_can_end_with_agents(duplicate_state, [agent, random_agent], rs), "Game did not end in 10000 turns. Random seed: " + str(rs) + ". Agent: " + str(agent) + "as first player"
    return True

def produce_agent(name):
    if name=="random": return arand.RandomPlayer()
    elif name=="mcts": return mcts.MCTS_Player(max_iterations=100, logs=True)
    elif name=="siea_mcts": return siea_mcts.SIEA_MCTS_Player(max_iterations=100, logs=True)
    elif name == "siea_mcts_unpaired": return siea_mcts.SIEA_MCTS_Player(max_iterations=100, logs=True, unpaired_evolution=True, name="siea_mcts_unpaired")
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

def test_fm_calls(iterations = 20):
    a1 = mcts.MCTS_Player(max_iterations=-1, rollouts = 1, logs=True)
    a2 = siea_mcts.SIEA_MCTS_Player(max_iterations=-1, rollouts = 1, logs=True)
    g = mnk.GameState(3,3,3)
    g.set_initial_state()
    a1.choose_action(g)
    a2.choose_action(g)
    for a in [a1,a2]:
        a._update_choose_action_logs()
        print(a.choose_action_logs)
    for a in [a1,a2]:
        for i in range(iterations):
            a.iteration(a.root_node)
            a._update_choose_action_logs()
        print("agent:" + a.name)
        print("fm calls: " + str(a.choose_action_logs["forward_model_calls"]))
        print("tree:" + a.view_mcts_tree())
        print("expanded_nodes: " + str(len(a.root_node.subtree_nodes())))

def test_action_logs(agent, state):
    agent.choose_action(state)
    agent.choose_action_logs

def test_fo_single_decision(logs_path = ""):
    function_index = 1
    runs = 3
    random_seed = 1
    iterations = 50
    game_state = fo.GameState(function_index=function_index)
    game_state.set_initial_state()
    mcts_player = siea_mcts.SIEA_MCTS_Player(max_iterations=iterations, logs=True)
    #mcts_player = mcts.MCTS_Player(max_iterations=iterations, logs=True)
    action = eu.mcts_decision_analysis(game_state, 
                                    mcts_player, 
                                    os.path.join(logs_path, "fo_single_decision_", mcts_player.name), 
                                    runs, 
                                    random_seed)
    mcts_player = mcts.MCTS_Player(max_iterations=iterations, logs=True)
    #mcts_player = mcts.MCTS_Player(max_iterations=iterations, logs=True)
    action = eu.mcts_decision_analysis(game_state, 
                                    mcts_player, 
                                    os.path.join(logs_path, "fo_single_decision_", mcts_player.name), 
                                    runs, 
                                    random_seed)

def test_tree_cloning():
    state = produce_game("mnk")
    agent = produce_agent("mcts")
    agent.max_iterations = 100
    _ = agent.choose_action(state)
    node_clone = agent.root_node.duplicate()
    node_clone.update(1)
    assert agent.root_node.visits != node_clone.visits, "Node visits on the clone not different, both paired to the same node"
    assert agent.root_node.total_reward != node_clone.total_reward, "Node total reward on the clone not different, both paired to the same node"
    for key,child in agent.root_node.children.items():
        child.update(1)
        assert child.total_reward != node_clone.children[key].total_reward, "Child total reward on the clone not different, both paired to the same node"
        assert child.visits != node_clone.children[key].visits, "Child visits on the clone not different, both paired to the same node"

def run(game_names=None, agent_names=None):
    #Database
    if game_names is None: game_names = ["mnk", "fo1d1p", "fo1d2p", "fo2d1p", "fo2d2p"]
    if agent_names is None: agent_names = ["random", "mcts", "siea_mcts", "siea_mcts_unpaired"]

    identifier = rd.randint(1,2**10)
    print("Unit test identifier: " + str(identifier))
    logs_path = os.path.join("Outputs","Unit_test_outputs",str(identifier))

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
    test_game_player.play_games(n_games=4, logs=True, logs_path=logs_path)
    test_game_player.play_game()
    print("Play_game logs:")
    print(test_game_player.logs_by_game)
    print("Experiment utils tests passed")

    #Advanced tests
    print("Advanced tests running")
    print("FM calls tests running")
    test_fm_calls()

    #Test tree cloning
    print("Tree cloning tests running")
    test_tree_cloning()

    #Test MCTS tree FO single decision
    print("FO single decision tests running")
    test_fo_single_decision(logs_path=logs_path)


   