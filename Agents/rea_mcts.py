import operator
from operator import attrgetter
import numpy as np
import time
import random as rd
from statistics import mean
import math
import os
import pandas as pd
import Agents.vanilla_mcts as vmcts# import MCTS_Player, Node
from Agents.random import RandomPlayer
import statistics as st
import Utilities.logs_management as lm
import Utilities.experiment_utils as eu

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

#ea and siea dont increase their iterations when they evolve

# want to maximise the solution
creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #This can be outside
# define the structure of the programs 
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)  #This can be outside

#create dummy trees. The first child should be preferred
dummy_action_names = ["I should win", "I should lose"]
class DummyState:
    def __init__(self, player_turn=None):
        self.player_turn = player_turn
        self.is_terminal = False
dummystate = DummyState()
dummy_root_node1 = vmcts.Node(parent = None, state=dummystate, expansion_index=0)
child1 = dummy_root_node1.add_child(edge_content=dummy_action_names[0], state=None, expansion_index=1)
child1.update(new_reward=1)
dummy_root_node1.update(new_reward=1)
child2 = dummy_root_node1.add_child(edge_content=dummy_action_names[1], state=None, expansion_index=2)
child2.update(new_reward=0)
dummy_root_node1.update(new_reward=0)

dummy_root_node2 = vmcts.Node(parent = None, state=dummystate, expansion_index=0)
child1 = dummy_root_node2.add_child(edge_content=dummy_action_names[0], state=None, expansion_index=1)
child1.update(new_reward=0)
dummy_root_node2.update(new_reward=0)
child2 = dummy_root_node2.add_child(edge_content=dummy_action_names[1], state=None, expansion_index=2)
child2.update(new_reward=0)
dummy_root_node2.update(new_reward=0)
child2.update(new_reward=0)
dummy_root_node2.update(new_reward=0)

test_trees = [dummy_root_node1, dummy_root_node2]

class SIEA_MCTS_Player(vmcts.MCTS_Player):

    def __init__(self, 
                 rollouts=1, 
                 c=math.sqrt(2), 
                 max_fm=np.inf, 
                 max_time=np.inf, 
                 max_iterations=np.inf, 
                 default_policy = RandomPlayer(), 
                 name = None,
                    es_lambda = 4,
                    es_fitness_iterations = 30,
                    es_generations = 20, #only matters if parallel_evolution = False
                    es_semantics_l = 5, #only matters if use_semantics = True
                    es_semantics_u = 10, #only matters if use_semantics = True
                    #variants:
                    use_semantics = True,
                    parallel_evolution = False, #-> mitigate feedback loop effect
                    #partial_evolution = False,
                    no_terminal_no_parent = False,
                    no_terminal_no_parent_terminals = ["Q","n"],
                    re_evaluation = False,
                logs_every_iterations = None,
                 logs = False):
        super().__init__(rollouts = rollouts, c=c, max_fm=max_fm, max_time=max_time, max_iterations=max_iterations, default_policy=default_policy, name=name, logs=logs, logs_every_iterations= logs_every_iterations)
        self.es_lambda = es_lambda
        self.es_fitness_iterations = es_fitness_iterations
        self.es_generations = es_generations
        self.es_semantics_l = es_semantics_l
        self.es_semantics_u = es_semantics_u
        self.use_semantics = use_semantics
        self.parallel_evolution = parallel_evolution
        #self.partial_evolution = partial_evolution
        self.no_terminal_no_parent = no_terminal_no_parent
        self.no_terminal_no_parent_terminals = no_terminal_no_parent_terminals
        self.re_evaluation = re_evaluation

        self.GPTree = None
        self.hasGPTree = False
        self.evolution_logs = pd.DataFrame()
        self.choose_action_logs = pd.DataFrame()
        self.isAIPlayer = True

        if name is None:
            self.name = "REA_MCTS__"
            self.name = self.name + "PE_" + str(self.parallel_evolution) + "__"
            #self.name = self.name + "PA_" + str(self.partial_evolution) + "__"
            self.name = self.name + "NP_" + str(self.no_terminal_no_parent) + "__"
            self.name = self.name + "RE_" + str(self.re_evaluation)

    def choose_action(self, state):
        self.evolution_logs = pd.DataFrame()
        self.final_evolution_data = pd.DataFrame()
        self.hasGPTree = False
        self.GPTree = None
        self.evolution_fm_calls = 0
        to_return = super().choose_action(state)
        return to_return

    def selection(self, node, my_tree_policy_formula=None, allow_evolution = True):
        #Returns a node that can be expanded selecting by UCT
        assert node.state.is_terminal == False, "Selection called on a terminal node"
        if not node.can_be_expanded() and not node.state.is_terminal:
            if not self.hasGPTree and allow_evolution: 
                self.GPTree = ES_Search(node, self)
                #print("Finished evolving")
                self.hasGPTree = True
            #node = self.best_child_by_tree_policy(node.children.values())
            #self.current_fm = self.current_fm + 1
            node = super().selection(node, my_tree_policy_formula=my_tree_policy_formula)
        return node
    
    def iteration(self, node=None):

        if node is None:
            node = self.root_node

        #Selection
        node = self.selection(node) #this will always return a decision node
        
        if self.parallel_evolution:
            if self.stopping_criteria():
                return

        #Expansion
        if node.can_be_expanded():
            node = self.expansion(node)

        #Simulation
        reward = self.simulation(node, self.rollouts, self.default_policy) 

        #Backpropagation
        self.backpropagation(node, reward)

        self.current_iterations = self.current_iterations + 1

    def tree_policy_formula(self, node):
        if not self.hasGPTree:
            return super().tree_policy_formula(node) #UCB1
        #Return Evolved formula
        else:
            if node.visits == 0:
                return np.inf
            if node.parent.state.player_turn == self.player:
                return self.GPTree(node.average_reward(), node.visits, node.parent.visits) 
            else:
                return -self.GPTree(node.average_reward(), node.visits, node.parent.visits)

    def _update_evolution_logs(self, data):
        #Data us a dataframe.
        self.evolution_logs = pd.concat([self.evolution_logs, data], ignore_index=True)

    def agent_data(self):
        agent_data = super().agent_data()
        data_dict = {
            "es_lambda":self.es_lambda,
            "es_fitness_iterations":self.es_fitness_iterations,
            "es_generations":self.es_generations,
            "es_semantics_l":self.es_semantics_l,
            "es_semantics_u":self.es_semantics_u,
            "use_semantics":self.use_semantics,
            "parallel_evolution":self.parallel_evolution,
            "no_terminal_no_parent":self.no_terminal_no_parent,
            "no_terminal_no_parent_terminals":str(self.no_terminal_no_parent_terminals),
            "re_evaluation":self.re_evaluation,
        }
        data_df = pd.DataFrame(data_dict, index=[0])
        return pd.concat([agent_data, data_df], axis=1)

    def _update_choose_action_logs(self):
        super()._update_choose_action_logs()
        assert "forward_model_calls" in self.choose_action_logs.columns, "Forward model calls not in choose_action_logs"
        self.choose_action_logs["evolved_a_tree"] = self.hasGPTree

        evolution_data = {}
        if self.hasGPTree:
            self.choose_action_logs["evolved_formula_str"] = str(self.GPTree)
            evolution_data["evolution_fm_calls"] = self.evolution_fm_calls
            evolution_data["avg_total_nodes"] = st.mean(self.evolution_logs["total_nodes"])
            evolution_data["std_total_nodes"] = st.stdev(self.evolution_logs["total_nodes"])
            evolution_data["avg_average_nodes"] = st.mean(self.evolution_logs["average_nodes"])
            evolution_data["std_average_nodes"] = st.stdev(self.evolution_logs["average_nodes"])
            evolution_data["avg_average_depth"] = st.mean(self.evolution_logs["average_depth"])
            evolution_data["std_average_depth"] = st.stdev(self.evolution_logs["average_depth"])
            evolution_data["avg_average_SSD"] = st.mean(self.evolution_logs["average_SSD"])
            valid_ssds = True
            for ssd in self.evolution_logs["average_SSD"]:
                if ssd == np.inf or ssd == -np.inf:
                    evolution_data["avg_average_SSD"] = np.inf
                    valid_ssds = False
                    break
            if valid_ssds:
                evolution_data["std_average_SSD"] = st.stdev(self.evolution_logs["average_SSD"]) if len(self.evolution_logs["average_SSD"]) > 1 else None
            collected_fitnesses = []
            for c in self.evolution_logs.columns:
                if "fitness_individual" in c:
                    collected_fitnesses = collected_fitnesses + list(self.evolution_logs[c])
            evolution_data["avg_fitness"] = st.mean(collected_fitnesses)
            evolution_data["std_fitness"] = st.stdev(collected_fitnesses)
            evolution_data["semantics_usage_ratio"] = st.mean(self.evolution_logs["semantics_used"])
            semantics_chose_randomly_filtered = [x for x in self.evolution_logs["semantics_chose_randomly"] if x is not None]
            evolution_data["semantics_chose_randomly_ratio"] = st.mean(semantics_chose_randomly_filtered) if len(semantics_chose_randomly_filtered) > 0 else None

            evolution_df = pd.DataFrame(evolution_data, index=[0])
            self.choose_action_logs = pd.concat([self.choose_action_logs, evolution_df], axis=1)
            self.choose_action_logs = pd.concat([self.choose_action_logs, self.final_evolution_data], axis=1)

    def dump_my_logs(self, path):
        super().dump_my_logs(path)
        lm.dump_data(self.evolution_logs, path, "evolution_logs.csv")

def randomC():
    c = rd.choice([0.25, 0.5, 1, 2, 3, 5, 7, 10])
    return c

def ES_Search(RootNode, MCTS_Player):
    """
    Find the best child from the given node
    """
    # initialize state variables
    state = RootNode.state  # current game state
    turn = state.turn
    
    # initialize MCTS_Player variables
    es_lambda = MCTS_Player.es_lambda
    es_generations = MCTS_Player.es_generations
    es_fitness_iterations = MCTS_Player.es_fitness_iterations
    hasGPTree = MCTS_Player.hasGPTree
    GPTree = MCTS_Player.GPTree
    
    # set the number of inputs - [Q,n,N,c]
    pset = gp.PrimitiveSet("MAIN", 3)
    
    # Define new functions
    def div(left, right):
        if (abs(right) < 0.001):
            return 1
        else:
            return left/right
        
    # natural log
    def ln(left): 
        if left == 1: left = 1.001
        if left < 0.01: left = 0.01
        return np.log(abs(left))
    
    # square root
    def root(left):
        return (abs(left))**(1/2)

    # add operators
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(div, 2)
    pset.addPrimitive(ln, 1)
    pset.addPrimitive(root, 1)
    #pset.addPrimitive(operator.neg, 1)

    # rename the arguments
    random_C= randomC()
    pset.addTerminal(random_C, name='c')
    pset.addTerminal(2)
    pset.renameArguments(ARG0='Q')
    pset.renameArguments(ARG1='n')
    pset.renameArguments(ARG2='N')
    pset.renameArguments(ARG3='c')
    
    # primitives and terminals list
    prims = pset.primitives[object]
    terminals = pset.terminals[object]
    #t0 = Q, t1 = n, t2 = N, t3 = c, t4=2
    
    #  register the generation functions into a Toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def evalTree(individual, RootNode, mcts_player): #OK
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)

        def my_tree_policy_formula(node):
                if node.visits == 0:
                    return np.inf
                if node.parent.state.player_turn == mcts_player.player:
                    return func(node.average_reward(), node.visits, node.parent.visits) 
                else:
                    return func(-node.average_reward(), node.visits, node.parent.visits)

        results = []
        for i in range(es_fitness_iterations):

            #Stop evolution
            if mcts_player.stopping_criteria():
                individual.semantics = sorted(results)
                return 0,

            node = RootNode
            
            #Selection
            node = mcts_player.selection(node, my_tree_policy_formula=my_tree_policy_formula, allow_evolution=False)

            if node.can_be_expanded():
                node = mcts_player.expansion(node)
            
            # simulation
            previous_fm = mcts_player.current_fm
            reward = mcts_player.simulation(node, mcts_player.rollouts, mcts_player.default_policy)
            mcts_player.evolution_fm_calls = mcts_player.evolution_fm_calls + (mcts_player.current_fm - previous_fm)
            results.append(reward)
            
            #Backpropogate
            mcts_player.backpropagation(node, reward)

            mcts_player.current_iterations = mcts_player.current_iterations + 1

            #PARALLEL EVOLUTION
            if not mcts_player.stopping_criteria():
                if mcts_player.parallel_evolution:
                    node = RootNode
            
                    #Selection
                    node = mcts_player.selection(node, allow_evolution=False) #This selection is normal ucb1

                    if node.can_be_expanded():
                        node = mcts_player.expansion(node)
                    
                    # simulation
                    previous_fm = mcts_player.current_fm
                    reward = mcts_player.simulation(node, mcts_player.rollouts, mcts_player.default_policy)
                    mcts_player.evolution_fm_calls = mcts_player.evolution_fm_calls + (mcts_player.current_fm - previous_fm)
                    results.append(reward)
                    
                    #Backpropogate
                    mcts_player.backpropagation(node, reward)

                    mcts_player.current_iterations = mcts_player.current_iterations + 1
        
        # semantics check  
        individual.semantics = sorted(results)
        
        fitness = np.mean(results)
        
        return fitness,

    # register gp functions
    toolbox.register("evaluate", evalTree, RootNode=RootNode, mcts_player=MCTS_Player)
    toolbox.register("select", selBestCustom)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
    toolbox.register("mutate", mutUniformCustom, expr=toolbox.expr_mut, pset=pset)
    
    # max depth of 8
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
    
    # create tree for original UCT
    UCT_formula = [ prims[0], terminals[0], prims[2], prims[2], terminals[4], terminals[3], 
                   prims[5], prims[3], prims[2], terminals[4], prims[4], terminals[2], terminals[1]]
    UCT_GP_Tree = creator.Individual(UCT_formula)

    
    # if MCTS already has a gpTree, return the best child nodenode
    if hasGPTree:
        func = toolbox.compile(expr=GPTree)
        nodeValues = {a:func(c.total_reward, c.visits, RootNode.visits) for a,c in RootNode.children.items()}
        node = RootNode.children[max(nodeValues, key=nodeValues.get)] if MCTS_Player.player == RootNode.state.player_turn else RootNode.children[min(nodeValues, key=nodeValues.get)]
        return node
    
    # else, find the optimal tree using GP 
    else:
        #print("Evolving UCT formula...")
        pop = [UCT_GP_Tree]  # one formula in tree
        hof = tools.HallOfFame(1)
            
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        pop = eaMuCommaLambdaCustom(MCTS_Player, turn, pop, toolbox, 
                                             mu=1, lambda_=es_lambda, ngen=es_generations, pset=pset, cxpb=0, mutpb=1, 
                                             stats=mstats, halloffame=hof, verbose=False)
        # return the best tree
        formula = str(hof[0])
        
        # append data to csv
        data = {'evolved_formula_not_ucb1':(UCT_GP_Tree != hof[0]),
                'evolved_formula': formula, 
                'evolved_formula_nodes':len(hof[0]), 
                'evolved_formula_depth': (hof[0]).height,
                "evolved_c": random_C}
        #MCTS_Player._update_choose_action_logs(pd.DataFrame(data, index=[0])) 
        #MCTS_Player.choose_action_logs = pd.concat([MCTS_Player.choose_action_logs, pd.DataFrame(data, index=[0])], axis=1)
        MCTS_Player.final_evolution_data = pd.DataFrame(data, index=[0])
        
        print(f'Chosen formula: {formula}')
        return toolbox.compile(expr=hof[0])
        #return toolbox.compile(UCT_GP_Tree)

#################################################################################  

def eaMuCommaLambdaCustom(MCTS_Player, turn, population, toolbox, mu, lambda_, ngen, pset, cxpb, mutpb, stats=None, halloffame=None, verbose=__debug__): #OK
    """
    This is the :math:`(\mu~,~\lambda)` evolutionary algorithm
    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    gen = 1
    while not MCTS_Player.stopping_criteria():
        gen += 1
        #print("In generation: " + str(gen) + " of " + str(ngen) + "...")
        # Vary the population
        #print("Generating offsprings")
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb, pset, MCTS_Player)

        # Evaluate the individuals with an invalid fitness
        #print("Evaluating individuals")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if MCTS_Player.re_evaluation: 
            invalid_ind = invalid_ind + population #adds the parent to be evaluated again
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        #print("Fitnesses: " + str(list(fitnesses)))
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            if MCTS_Player.re_evaluation:
                halloffame.update(population + offspring)
            else: halloffame.update(offspring)


        # Select the next generation population
        #print("Selecting next generation, pop size:", str(len(population)), ", offspring size:",str(len(offspring)))
        population[:] = toolbox.select(population + offspring, MCTS_Player, gen, turn) #MU PLUS LAMBDA

        # Update the statistics with the new population
        
        if verbose:
            #print("Updating statistics")
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            #print(logbook.stream)
        
        if not MCTS_Player.parallel_evolution and gen == ngen+1:
            break

    return population

def semanticsDistance(original, new): #OK
    return sum((np.absolute(np.subtract(original.semantics, new.semantics))/len(new.semantics)))

def selBestCustom(individuals, MCTS_Player, generation, turn, fit_attr="fitness"): #OK
    # initialize values to add to table
    Nodes = 0  # total number of nodes
    SSD = 0 # total semantic distance
    TotalDepth = 0  # add depth of each indivdual
    L = MCTS_Player.es_semantics_l
    U = MCTS_Player.es_semantics_u
    
    numInd = len(individuals)
    
    # keep track of fitness and semantic values
    fitnesses_list = []
    SSD_list = []
    
    # iterate through each individual program
    #print("Calculating SD and depth")
    for i in individuals:
        Nodes += len(i)  # number of nodes in each individual
        # SSD between each new individual and sole parent
        if len(i.semantics) == len(individuals[0].semantics):  
            distance = round(semanticsDistance(individuals[0], i), 3)
        else: distance = np.inf
        i.SD = distance
        SSD += distance
        TotalDepth += i.height
        # append to lists
        

        fitnesses_list.append(i.fitness.values)
        SSD_list.append(distance)
        
    # check how many equal max fitness
    numberMax = fitnesses_list.count(max(fitnesses_list))
    #print("Number of max fitnesses: " + str(numberMax))
    
    bestIndex = None
    isRandom = None
    semantics_used = False
    if numberMax > 1 and MCTS_Player.use_semantics:
        semantics_used = True
        bestIndex, isRandom = SemanticsDecider(fitnesses_list, SSD_list, MCTS_Player, turn, generation,L,U)
        to_return = [individuals[bestIndex]]
    else:
        # sorted by fitness
        ind_sorted = sorted(individuals, key=attrgetter("fitness"), reverse=True)
        to_return = ind_sorted[:1]
    

    #Update evolution file
    data = {'generation':generation, 
            'es_lambda':numInd-1, 
            'total_nodes': Nodes, 
            'average_nodes':Nodes/numInd, 
            'average_depth': TotalDepth/(numInd), 
            'average_SSD':SSD/(numInd-1),
            "semantics_used": semantics_used,
            'best_index_by_semantics':bestIndex, 
            'semantics_chose_randomly':isRandom,
            "semantics_L": L,
            "semantics_U": U,
            "evolution_fm_calls":MCTS_Player.evolution_fm_calls
            }
    for i,v in enumerate(fitnesses_list):
        data['fitness_individual'+str(i)] = v
    for i,v in enumerate(SSD_list):
        data['SSD_individual'+str(i)] = v
    for i,ind in enumerate(individuals):
        data['individual'+str(i)] = str(ind)
    MCTS_Player._update_evolution_logs(pd.DataFrame(data, index=[0]))

    return to_return
    
def SemanticsDecider(fitnesses_list, SSD_list, MCTS_Player, turn, generation, L, U): #OK
    #print(f'(ES Semantics) Fitness: {fitnesses_list}, SSD: {SSD_list}')
    # lower and upper thresholdof SSD
    
    # index of max values
    indices = [i for i, x in enumerate(fitnesses_list) if x == max(fitnesses_list)]
    
    isRandom = False
    lowest = U
    bestIndex = None
    
    for i in indices:
        if L < SSD_list[i] < lowest:
            lowest = SSD_list[i]  # new lowest value
            bestIndex = i  # new best index
    
    # if none match criteria
    if bestIndex is None:
        isRandom = True
        bestIndex = rd.choice(indices)
    
    # return best index of best individual
    #print(f'(ES Semantics) Best Index: {bestIndex}')
    return bestIndex, isRandom
        
def varOr(population, toolbox, lambda_, cxpb, mutpb, pset, MCTS_Player): #OK
    """
    Part of an evolutionary algorithm applying only the variation part
    (crossover, mutation **or** reproduction). The modified individuals have
    their fitness invalidated. The individuals are cloned so returned
    population is independent of the input population.
    """
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        # randomly change c parameter
        pset.terminals[object][3] = gp.Terminal(randomC(), True, object)
        pset.renameArguments(ARG3='c')
        
        
        #ind, = toolbox.mutate(ind)
        # make sure it's a new program less than or equal to depth 8
        
        #while((ind == population[0]) or (ind.height > 8) ):
        #    ind, = toolbox.mutate(ind)

        offspring_not_valid = True
        not_valid_count = 0
        while offspring_not_valid:
            not_valid_count += 1
            continue_loop = False

            #generate offspring
            ind = toolbox.clone(rd.choice(population))
            ind, = toolbox.mutate(ind)

            #check validity
            if ind.height > 8:
                continue_loop = True
            if ind == population[0]:
                continue_loop = True

            if MCTS_Player.no_terminal_no_parent: #perform this test only if no_terminal_no_parent is true
                func = toolbox.compile(expr=ind)

                def my_tree_policy_formula(node):
                    if node.visits == 0:
                        return np.inf
                    if node.parent.state.player_turn == MCTS_Player.player:
                        return func(node.average_reward(), node.visits, node.parent.visits) 
                    else:
                        return func(-node.average_reward(), node.visits, node.parent.visits)
                
                for test_tree in test_trees:
                    dummy_values = {ea:None for ea in dummy_action_names}
                    test_tree.state.player_turn = MCTS_Player.root_node.state.player_turn
                    for edge_action in dummy_action_names:
                        #test_tree.children[edge_action].state.player_turn = RootNode.state.player_turn
                        dummy_values[edge_action] = my_tree_policy_formula(test_tree.children[edge_action])
                    if dummy_values[dummy_action_names[0]] <= dummy_values[dummy_action_names[1]]:
                        #individual is not valid
                        continue_loop = True

            #if we reach this point, the individual is valid
            if not continue_loop:
                offspring_not_valid = False

            #check if we are stuck in an infinite loop
            if not_valid_count > 1000:
                print("Over 1000 invalid individuals generated")
                break
        #if not_valid_count > 2:
        #    print("Generated offspring in " + str(not_valid_count) + " tries")
    
        del ind.fitness.values
        offspring.append(ind)
        
    return offspring

def mutUniformCustom(individual, expr, pset): #OK
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.
    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    
    # Mutation:
    #   90% internal nodes
    #   10% leaves
    
    isLeaf = (rd.random() > 0.9)  # random
    
    if isLeaf:
        # get index of terminals
        isTerminals = [node in pset.terminals[object] for node in individual]
        indices = [i for i, x in enumerate(isTerminals) if x]
    else:
        # get index of primitives
        isPrims = [node in pset.primitives[object] for node in individual]
        indices = [i for i, x in enumerate(isPrims) if x]
    
    if indices == []:
        index = 0
    else:
        index = rd.choice(indices)  # choose random index
        
    slice_ = individual.searchSubtree(index)  # cut individual at chosen index
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)

    return individual,






"""
if MCTS_Player.no_terminal_no_parent: #test if formula does the bare minimum to be valid
            #f = str(i)
            #for terminal in MCTS_Player.no_terminal_no_parent_terminals:
            #    appearances = f.count(terminal) ##ATTENTION: the terminals are found on the strings, might require particular ways to find them (like "n" mixed with the operand ln)
            #    if terminal == "n":
            #        appearances -= f.count("ln")
            #    if appearances == 0:
            #        #individual is not valid
            #        i.fitness.values = -99999,
            #        break
            for test_tree in test_trees:
                dummy_values = {ea:None for ea in dummy_action_names}
                test_tree.state.player_turn = RootNode.state.player_turn
                for edge_action in dummy_action_names:
                    #test_tree.children[edge_action].state.player_turn = RootNode.state.player_turn
                    dummy_values[edge_action] = my_tree_policy_formula(test_tree.children[edge_action])
                if dummy_values[dummy_action_names[0]] <= dummy_values[dummy_action_names[1]]:
                    #individual is not valid
                    individual.semantics = [None]
                    return -99999,


"""
