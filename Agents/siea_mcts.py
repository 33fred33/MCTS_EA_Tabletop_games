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

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp



class SIEA_MCTS_Player(MCTS_Player):

    def __init__(self, 
                 rollouts=1, 
                 c=math.sqrt(2), 
                 max_fm=np.inf, 
                 max_time=np.inf, 
                 max_iterations=np.inf, 
                 default_policy = RandomPlayer(), 
                 name = "SIEA_MCTS",
                    es_lambda = 4,
                    es_fitness_iterations = 30,
                    es_generations = 20,
                    es_semantics_l = 5,
                    es_semantics_u = 10,
                 logs = False):
        super().__init__(rollouts, c, max_fm, max_time, max_iterations, default_policy, name, logs)
        self.es_lambda = es_lambda
        self.es_fitness_iterations = es_fitness_iterations
        self.es_generations = es_generations
        self.es_semantics_l = es_semantics_l
        self.es_semantics_u = es_semantics_u

        self.GPTree = None
        self.hasGPTree = False
        self.evolution_logs = pd.DataFrame()
        self.choose_action_logs = pd.DataFrame()


    def choose_action(self, state):
        self.hasGPTree = False
        super().choose_action(state)

    def selection(self, node) -> Node:
        #Returns a node that can be expanded selecting by UCT
        assert node.state.is_terminal == False, "Selection called on a terminal node"
        while not node.can_be_expanded() and not node.state.is_terminal:
            if not self.hasGPTree: 
                self.GPTree = ES_Search(node, self)
                self.hasGPTree = True
            node = ES_Search(node, self)
            #node = max(node.children.values(), key= lambda x: self.UCB1(x))
        return node
    
    def _update_evolution_logs(self, data):
        #Data us a dataframe.
        self.evolution_logs = pd.concat([self.evolution_logs, data], ignore_index=True)

    def _update_choose_action_logs(self, data):
        #Data is a dataframe, singlerow.
        self.choose_action_logs = pd.concat([self.choose_action_logs, data], axis=1)


# random constant
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
    pset.addTerminal(randomC(), name='c')
    pset.addTerminal(2)
    pset.renameArguments(ARG0='Q')
    pset.renameArguments(ARG1='n')
    pset.renameArguments(ARG2='N')
    pset.renameArguments(ARG3='c')
    
    # primitives and terminals list
    prims = pset.primitives[object]
    terminals = pset.terminals[object]
    
    # want to maximise the solution
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    # define the structure of the programs 
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax) 
    
    #  register the generation functions into a Toolbox
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    def evalTree(individual, RootNode, mcts_player): #OK
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        
        # from this point simulate the game 10 times appending the results
        results = []
        for i in range(es_fitness_iterations):
            # copy the state
            stateCopy = RootNode.state.duplicate()
            node = RootNode
            
            # child nodes
            #v =  [func(Q,n,N) for Q,n,N in nodeValues]
            nodeValues = {a:func(c.total_reward, c.visits, node.visits) for a,c in node.children.items()} # values of the nodes
            # get the values of the tree for each child node
            
            node = node.children[max(nodeValues, key=nodeValues.get)] if mcts_player.player == node.state.player_turn else node.children[min(nodeValues, key=nodeValues.get)]
            
            # play the move of this child node
            stateCopy.make_action(node.edge_action)
            
            # random rollout
            while not stateCopy.is_terminal:
                stateCopy.make_action(mcts_player.default_policy.choose_action(stateCopy))
                #stateCopy.make_action()
                
            # result
            reward = stateCopy.reward[mcts_player.player]
            results.append(reward)
            
            #Backpropogate
            while node != None:  # backpropogate from the expected node and work back until reaches root_node
                node.update(new_reward = reward)
                node = node.parent
        
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

    
    # if MCTS already has a gpTree, return the best child node
    if hasGPTree:
        func = toolbox.compile(expr=GPTree)
        nodeValues = {a:func(c.total_reward, c.visits, RootNode.visits) for a,c in RootNode.children.items()}
        node = node.children[max(nodeValues, key=nodeValues.get)] if MCTS_Player.player == node.state.player_turn else node.children[min(nodeValues, key=nodeValues.get)]
        return node
    
    # else, find the optimal tree using GP 
    else:
        print("Evolving UCT formula...")
        pop = [UCT_GP_Tree]  # one formula in tree
        hof = tools.HallOfFame(1)
            
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        
        pop, logbook = eaMuCommaLambdaCustom(MCTS_Player, turn, pop, toolbox, 
                                             mu=1, lambda_=es_lambda, ngen=es_generations, pset=pset, cxpb=0, mutpb=1, 
                                             stats=mstats, halloffame=hof, verbose=False)
        # return the best tree
        formula = str(hof[0])
        
        # append data to csv
        data = {'evolved_formula_not_ucb1':(UCT_GP_Tree != hof[0]), 
                'evolved_formula': formula, 
                'evolved_formula_nodex':len(hof[0]), 
                'evolved_formula_depth': (hof[0]).height }
        MCTS_Player._update_choose_action_logs(pd.Dataframe(data), index=[0]) 
        
        #print(f'Chosen formula: {formula}')
        return hof[0]

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
    for gen in range(1, ngen + 1):
        print("In generation: " + str(gen) + " of " + str(ngen) + "...")
        # Vary the population
        print("Generating offsprings")
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb, pset)

        # Evaluate the individuals with an invalid fitness
        print("Evaluating individuals")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        print("Firnesses: " + str(fitnesses))
        
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        print("Selecting next generation, pop size:", str(len(population)), ", offspring size:",str(len(offspring)))
        population[:] = toolbox.select(population + offspring, MCTS_Player, gen, turn)

        # Update the statistics with the new population
        print("Updating statistics")
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
    return population, logbook

def semanticsDistance(original, new): #OK
    return sum((np.absolute(np.subtract(original.semantics, new.semantics))/len(new.semantics)))

def selBestCustom(individuals, MCTS_Player, generation, turn, fit_attr="fitness"): #OK
    # initialize values to add to table
    Nodes = 0  # total number of nodes
    SSD = 0 # total semantic distance
    TotalDepth = 0  # add depth of each indivdual
    
    numInd = len(individuals)
    
    # keep track of fitness and semantic values
    fitnesses_list = []
    SSD_list = []
    
    # iterate through each individual program
    print("Calculating SD and depth")
    for i in individuals:
        Nodes += len(i)  # number of nodes in each individual
        # SSD between each new individual and sole parent
        distance = round(semanticsDistance(individuals[0], i), 3)
        i.SD = distance
        SSD += distance
        TotalDepth += i.height
        # append to lists
        fitnesses_list.append(i.fitness.values)
        SSD_list.append(distance)
        
    # check how many equal max fitness
    numberMax = fitnesses_list.count(max(fitnesses_list))
    print("Number of max fitnesses: " + str(numberMax))
    
    bestIndex = None
    isRandom = None
    if numberMax > 1:
        bestIndex, isRandom = SemanticsDecider(fitnesses_list, SSD_list, MCTS_Player, turn, generation)
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
            'fitnesses':fitnesses_list, 
            'SSDs': SSD_list, 
            'best_index_by_semantics':bestIndex, 
            'semantics_chose_randomly':isRandom
            }
    print("Updating evolution file")
    MCTS_Player._update_evolution_logs(pd.DataFrame(data, index=[0]))

    return to_return
    
def SemanticsDecider(fitnesses_list, SSD_list, MCTS_Player, turn, generation): #OK
    #print(f'(ES Semantics) Fitness: {fitnesses_list}, SSD: {SSD_list}')
    # lower and upper thresholdof SSD
    L = 5
    U = 10
    
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
        
def varOr(population, toolbox, lambda_, cxpb, mutpb, pset): #OK
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
        
        ind = toolbox.clone(rd.choice(population))
        ind, = toolbox.mutate(ind)
        # make sure it's a new program less than or equal to depth 8
        while((ind == population[0]) or (ind.height > 8) ):
            ind, = toolbox.mutate(ind)
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


##########################################################################################################





