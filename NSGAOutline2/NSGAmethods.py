# uncompyle6 version 3.5.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.5 (default, Aug  7 2019, 00:51:29) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-39)]
# Embedded file name: C:\Users\depila\Desktop\Graduate Research\GeneticAlgorithims\CreateNSGA2\NSGAOutline\NSGAmethods.py
# Compiled at: 2021-04-09 02:21:50
"""This file is meanto contain the methods and classes used for NSGA-2 and its interface with Gnowee"""
import numpy as np

def runNSGA_generation(P_t, Q_t, problemParametersObject):
    numVariables = len(problemParametersObject.varType)
    objectiveFunctionlist = problemParametersObject.objective
    num_cont_int_bin_variables = len(problemParametersObject.lb)
    num_Features = num_cont_int_bin_variables
    var_range = []
    for k in range(0, num_cont_int_bin_variables):
        var_range += (problemParametersObject.lb[k], problemParametersObject.ub[k])

    num_children = len(Q_t)
    child_list = []
    for k in range(0, num_children):
        child = pop_member(numVariables, var_range, features=Q_t[k] * np.ones((1, 1)))
        child.objectives = objectiveFunctionlist
        child_list.append(child)

    num_objectives = len(objectiveFunctionlist)
    for k in range(0, num_children):
        child_list[k].fitness = np.resize(child_list[k].fitness, (num_objectives, 1))

    for k in range(0, num_children):
        #print 'k = ', k
        for i in range(0, num_objectives):
            #print 'i = ', i
            child_list[k].fitness[i] = problemParametersObject.objective[i].func(Q_t[k])

        child_list[k].Evaluated = 1

    num_Parents = len(P_t)
    parent_list = []
    for k in range(0, num_Parents):
        parent = pop_member(numVariables, var_range, features=P_t[k].variables, fitness=P_t[k].fitness, changeCount=P_t[k].changeCount, stallCount=P_t[k].stallCount, Evaluated=1)
        parent.objectives = objectiveFunctionlist
        parent_list.append(parent)

    R_t = child_list + parent_list
    N = len(P_t)
    sorted_population = FastNonDomSort_Gnowee(R_t, N, problemParametersObject)
    new_population = Population()
    front_index = 0
    while len(new_population) + len(sorted_population.fronts[front_index]) <= N:
        eval_front = get_fronts_crowding_distances(sorted_population.fronts[front_index])
        new_population.extend(eval_front)
        front_index = front_index + 1

    eval_front = get_fronts_crowding_distances(sorted_population.fronts[front_index])
    eval_front.sort(key=lambda individual: individual.crowding_distance, reverse=True)
    new_population.extend(eval_front[0:N - len(new_population)])
    return new_population


def get_fronts_crowding_distances(front):
    if len(front) > 0:
        solutions_num = len(front)
        for individual in front:
            individual.crowding_distance = 0

        for m in range(len(front[0].objectives)):
            front.sort(key=lambda individual: individual.fitness[m])
            front[0].crowding_distance = 1000000000
            front[(solutions_num - 1)].crowding_distance = 1000000000
            m_values = [ individual.fitness[m] for individual in front ]
            scale = max(m_values) - min(m_values)
            if scale == 0:
                scale = 1
            for i in range(1, solutions_num - 1):
                front[i].crowding_distance += (front[(i + 1)].fitness[m] - front[(i - 1)].fitness[m]) / scale

    return front


def FastNonDomSort_Gnowee(R_t, N, ProblemParametersObject):
    """Takes in the combined parent and offspring population as arrays of EVALUATED pop_members.

    R_t is an array of pop_members

    Returns a population class with fronts"""
    fronts = [[]]
    num_total_initial = len(R_t)
    population = Population()
    for i in range(0, num_total_initial):
        R_t[i].objectives = ProblemParametersObject.objective
        population.append(R_t[i])

    #print 'Test'
    for individual in population:
        individual.domination_count = 0
        individual.dominated_solutions = []
        for other_individual in population:
            if individual.dominates(other_individual):
                individual.dominated_solutions.append(other_individual)
            elif other_individual.dominates(individual):
                individual.domination_count += 1

        if individual.domination_count == 0:
            individual.rank = 0
            population.fronts[0].append(individual)

    i = 0
    while len(population.fronts[i]) > 0:
        temp = []
        for individual in population.fronts[i]:
            for other_individual in individual.dominated_solutions:
                other_individual.domination_count -= 1
                if other_individual.domination_count == 0:
                    other_individual.rank = i + 1
                    temp.append(other_individual)

        i = i + 1
        population.fronts.append(temp)

    return population


class pop_member(object):

    def __init__(self, num_variables, variable_bounds, features=[], fitness=1000000000000000.0 * np.ones((1, 1)), changeCount=0, stallCount=0, Evaluated=0):
        self._num_variables = num_variables
        self._variable_bounds = variable_bounds
        self.features = features
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.objectives = None
        self.Evaluated = Evaluated
        self.variables = features
        self.fitness = fitness
        self.changeCount = changeCount
        self.stallCount = stallCount
        return

    def __str__(self):
        """Returns a string representation of the particular solution."""
        return self._values

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        num_objectives = len(self.fitness)
        first_fitness_set = np.zeros((num_objectives, 1))
        Second_fitness_set = np.zeros((num_objectives, 1))
        for p in range(0, num_objectives):
            first_fitness_set[p] = self.fitness[p]
            Second_fitness_set[p] = other_individual.fitness[p]

        for first, second in zip(self.fitness, other_individual.fitness):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second

        return and_condition and or_condition


class Population:

    def __init__(self):
        self.population = []
        self.fronts = [[]]

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)


class ObjectiveFunction(object):

    def __init__(self, input_vector, func):
        """this constructor is used to define the inputs to an objective function.

        input_vector is a 1 by num_variables (where num_variables includes every element in each solution)
        numpy array containing zeros coresponding to 
        non-inputs and 1 coreponding to inputs. The order of the zeros and ones is determined
        by the form the solutions are defined in.
        
        func is the basic function that takes in used_num_inputs inputs in the order they
        appear in input_vector (and solution object) from left to right"""
        self._input_vector = input_vector
        self._func = func
        self.used_num_inputs = np.sum(input_vector)

    def evaluate(solution_input):
        """the basic use of the Objective Function Class.
        The output is the fittness.

        solutino_input is an instance of a solution object that is to be evaluated using the objective function"""
        num_vars = solution_input._num_variables
        var_vec = np.zeros(self.used_num_inputs)
        counter = 0
        for n in range(num_vars):
            if self._input_vector[n] == 1:
                var_vec[counter] = solution_input._values[counter]
                counter = counter + 1

        value = self._func(*var_vec)
        return value
