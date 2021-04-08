# uncompyle6 version 3.5.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.5 (default, Aug  7 2019, 00:51:29) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-39)]
# Embedded file name: C:\Users\depila\Desktop\Graduate Research\GeneticAlgorithims\CreateNSGA2\NSGAOutline\GnoweeHeuristics_multi.py
# Compiled at: 2021-03-31 01:46:16
"""!
@file src/GnoweeHeuristics.py
@package Gnowee

@defgroup GnoweeHeuristics GnoweeHeuristics

@brief Heuristics and settings supporting the Gnowee metaheuristic optimization
algorithm.

This instantiates the class and methods necessary to perform an optimization
using the Gnowee algorithm.  Each of the heuristics can also be used
independently of the Gnowee algorithm by instantiating this class and choosing
the desired heuristic.

The default settings are those found to be best for a suite of benchmark
problems but one may find alternative settings are useful for the problem of
interest based on the fitness landscape and type of variables.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""
import numpy as np, copy as cp
from math import sqrt
from numpy.random import rand, permutation
from Sampling_multi import levy, tlf, initial_samples
from GnoweeUtilities_multi import ProblemParameters_multi, Event
from nsga2.problem import Problem
from nsga2.evolution import Evolution
from nsga2.individual import Individual
import random, NSGAmethods

class GnoweeHeuristics_multi(ProblemParameters_multi):
    """!
    @ingroup GnoweeHeuristics
    The class is the foundation of the Gnowee optimization algorithm.  It sets
    the settings required for the algorithm and defines the heurstics.
    """

    def __init__(self, population=25, initSampling='lhc', fracMutation=0.2, fracElite=0.2, fracLevy=1.0, alpha=0.5, gamma=1, n=1, scalingFactor=10.0, penalty=0.0, maxGens=20000, maxFevals=200000, convTol=1e-06, stallLimit=10000, optConvTol=0.01, **kwargs):
        r"""!
        Constructor to build the GnoweeHeuristics class.  This class must be
        fully instantiated to run the Gnowee program.  It consists of 2 main
        parts: The main class attributes and the inhereted ProblemParams class
        attributes.  The main class atrributes contain defaults that don't
        require direct user input to work (but can be modified by user input
        if desired), but the ProblemParameters class does require proper
        instantiation by the user.

        The default settings are found to be optimized for a wide range of
        problems, but can be changed to optimize performance for a particular
        problem type or class.  For more details, refer to the
        <a href='../docs/IEEE_Gnowee.pdf'>development paper</a>.

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param population: \e integer 

            The number of members in each generation. 

        @param initSampling: \e string 

            The method used to sample the phase space and create the initial
            population. Valid options are 'random', 'nolh', 'nolh-rp',
            'nolh-cdr', and 'lhc' as specified in init_samples(). 

        @param fracMutation : \e float 

            Discovery probability used for the mutate() heuristic. 

        @param fracElite: \e float 

            Elite fraction probability used for the scatter_search(),
            crossover(), and cont_crossover() heuristics. 

        @param fracLevy: \e float 

            Levy flight probability used for the disc_levy_flight() and
            cont_levy_flight() heuristics. 

        @param alpha: \e float 

            Levy exponent - defines the index of the distribution and controls
            scale properties of the stochastic process. 

        @param gamma: \e float 

            Gamma - scale unit of process for Levy flights. 

        @param n: \e integer 

            Number of independent variables - can be used to reduce Levy flight
            sampling variance. 

        @param penalty: \e float 

            Individual constraint violation penalty to add to objective
            function. 

        @param scalingFactor: \e float 

            Step size scaling factor used to adjust Levy flights to length scale
            of system. The implementation of the Levy flight sampling makes this
            largely arbitrary. 

        @param maxGens: \e integer 

            The maximum number of generations to search. 

        @param maxFevals: \e integer 

            The maximum number of objective function evaluations. 

        @param convTol: \e float 

            The minimum change of the best objective value before the search
            terminates. 

        @param stallLimit: \e integer 

            The maximum number of evaluations to search without an
            improvement. 

        @param optConvTol: \e float 

            The maximum deviation from the best know fitness value before the
            search terminates. 

        @param kwargs: <em> ProblemParameters class arguments </em> 

            Keyword arguments for the attributes of the ProblemParameters
            class. If not provided. The inhereted attributes will be set to the
            class defaults. 

        """
        ProblemParameters_multi.__init__(self, **kwargs)
        self.population = population
        self.initSampling = initSampling
        self.fracMutation = fracMutation
        assert self.fracMutation >= 0 and self.fracMutation <= 1, 'The probability of discovery must exist on (0,1]'
        self.fracElite = fracElite
        assert self.fracElite >= 0 and self.fracElite <= 1, 'The elitism fraction must exist on (0,1]'
        self.fracLevy = fracLevy
        assert self.fracLevy >= 0 and self.fracLevy <= 1, 'The probability that a Levy flight is performed must exist on (0,1]'
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.scalingFactor = scalingFactor
        self.penalty = penalty
        self.maxGens = maxGens
        self.maxFevals = maxFevals
        self.convTol = convTol
        self.stallLimit = stallLimit
        self.optConvTol = optConvTol

    def __repr__(self):
        """!
        GnoweeHeuristics print function.

        @param self: <em> GnoweeHeuristics pointer </em> 

            The GnoweeHeuristics pointer. 

        """
        return ('GnoweeHeuristics({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},                                  {}, {}, {}, {}, {})').format(self.population, self.initSampling, self.fracMutation, self.fracElite, self.fracLevy, self.alpha, self.gamma, self.n, self.scalingFactor, self.penalty, self.maxGens, self.maxFevals, self.convTol, self.stallLimit, self.optConvTol, ProblemParameters.__repr__())

    def __str__(self):
        """!
        Human readable GnoweeHeuristics print function.

        @param self: <em> GnoweeHeuristics pointer </em> 

            The GnoweeHeuristics pointer. 

        """
        header = [
         '  GnoweeHeuristics:']
        header += [('Population = {}').format(self.population)]
        header += [('Sampling Method = {}').format(self.initSampling)]
        header += [('Discovery Fraction = {}').format(self.fracMutation)]
        header += [('Elitism Fraction = {}').format(self.fracElite)]
        header += [('Levy Fraction = {}').format(self.fracLevy)]
        header += [('Levy Alpha = {}').format(self.alpha)]
        header += [('Levy Gamma = {}').format(self.gamma)]
        header += [('Levy Independent Samples = {}').format(self.n)]
        header += [('Levy Scaling Parameter = {}').format(self.scalingFactor)]
        header += [('Constraint Violaition Penalty = {}').format(self.penalty)]
        header += [('Max # of Generations = {}').format(self.maxGens)]
        header += [('Max # of Function Evaluations = {}').format(self.maxFevals)]
        header += [('Convergence Tolerance = {}').format(self.convTol)]
        header += [('Stall Limit = {}').format(self.stallLimit)]
        header += [('Optimal Convergence Tolerance = {}').format(self.optConvTol)]
        header += ['     Attributes Inhereted from ProblemParameters:']
        header += [('{}').format(ProblemParameters.__str__(self))]
        return ('\n').join(header) + '\n'

    def initialize(self, numSamples, sampleMethod):
        r"""!
        Initialize the population according to the sampling method chosen.

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param numSamples: \e integer 

            The number of samples to be generated. 

        @param sampleMethod: \e string 

            The method used to sample the phase space and create the initial
            population. Valid options are 'random', 'nolh', 'nolh-rp',
            'nolh-cdr', and 'lhc' as specified in init_samples(). 

        @return <em> list of arrays: </em> The initialized set of samples.
        """
        initSamples = initial_samples(self.lb, self.ub, sampleMethod, numSamples)
        if sum(self.xID) != 0:
            xUB = [
             self.ub[np.where(self.xID == 1)[0][0]]] * len(self.xID)
            xSamples = initial_samples([0] * len(self.xID), xUB, 'rand-wor', numSamples)
        for var in range(len(self.varType)):
            if 'i' in self.varType[var] or 'd' in self.varType[var]:
                initSamples[:, var] = np.rint(initSamples[:, var])

        if sum(self.xID) != 0:
            initSamples = initSamples * (self.cID + self.iID + self.dID) + xSamples * self.xID
        return initSamples

    def disc_levy_flight(self, pop):
        """!
        Generate new children using truncated Levy flights permutation of
        current generation design parameters according to:

        \x0c$ L_{\x07lpha,\\gamma}=FLOOR(TLF_{\x07lpha,\\gamma}*D(x)), \x0c$

        where \x0c$ TLF_{\x07lpha,\\gamma} \x0c$ is calculated in tlf(). Applies
        rejection_bounds() to ensure all solutions lie within the design
        space by adapting the step size to the size of the design space.

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \\e list: A list of the identities of the chosen index for
            each child.
        """
        children = []
        varID = self.iID + self.dID
        step = tlf(len(pop), len(pop[0]), alpha=self.alpha, gam=self.gamma)
        used = []
        for i in range(0, int(self.fracLevy * self.population), 1):
            k = int(rand() * self.population)
            while k in used:
                k = int(rand() * self.population)

            used.append(k)
            children.append(cp.deepcopy(pop[k]))
            stepSize = np.round(step[k, :] * varID * (self.ub - self.lb))
            if all(stepSize == 0):
                stepSize = np.round(rand(len(varID)) * (self.ub - self.lb)) * varID
            children[i] = (children[i] + stepSize) % (self.ub + 1 - self.lb)
            children[i] = rejection_bounds(pop[k], children[i], stepSize, self.lb, self.ub)

        return (
         children, used)

    def cont_levy_flight(self, pop):
        """!
        Generate new children using Levy flights permutation of current
        generation design parameters according to:

        \x0c$ x_r^{g+1}=x_r^{g}+ \x0crac{1}{\x08eta} L_{\x07lpha,\\gamma}, \x0c$

        where \x0c$ L_{\x07lpha,\\gamma} \x0c$ is calculated in levy() according
        to the Mantegna algorithm.  Applies rejection_bounds() to ensure all
        solutions lie within the design space by adapting the step size to
        the size of the design space.

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \\e list: A list of the identities of the chosen index for
            each child.
        """
        children = []
        varID = self.cID
        step = levy(len(pop[0]), len(pop), alpha=self.alpha, gam=self.gamma, n=self.n)
        used = []
        for i in range(int(self.fracLevy * self.population)):
            k = int(rand() * self.population)
            while k in used:
                k = int(rand() * self.population)

            used.append(k)
            children.append(cp.deepcopy(pop[k]))
            stepSize = 1.0 / self.scalingFactor * step[k, :] * varID
            children[i] = children[i] + stepSize
            children[i] = rejection_bounds(pop[k], children[i], stepSize, self.lb, self.ub)

        return (
         children, used)

    def comb_levy_flight(self, pop):
        r"""!
        Generate new children using truncated Levy flights permutation and
        inversion of current generation design parameters.

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """
        children = []
        used = []
        step = tlf(len(pop), len(pop[0]))
        for i in range(0, int(self.population * self.fracLevy)):
            k = int(rand() * self.population)
            while k in used:
                k = int(rand() * self.population)

            used.append(k)
            children.append(cp.deepcopy(pop[k]))
            tmp = [ children[(-1)][x] for x in range(0, len(self.xID)) if self.xID[x] == 1
                  ]
            for j in range(0, len(tmp) - 1):
                flight = (tmp[j] + int(step[i][j] * len(tmp))) % (len(tmp) - 1)
                if tmp[(j + 1)] != flight:
                    ind = np.where(tmp == flight)[0][0]
                    if ind > j:
                        tmp[(j + 1):(ind + 1)] = reversed(tmp[j + 1:ind + 1])
                    if j > ind:
                        tmp[ind:(j + 1)] = reversed(tmp[ind:j + 1])

            for x in range(0, len(self.xID)):
                if self.xID[x] == 1:
                    children[(-1)][x] = tmp[0]
                    del tmp[0]

        return (
         children, used)

    def scatter_search(self, pop):
        """!
        Generate new designs using the scatter search heuristic according to:

        \x0c$ x^{g+1} = c_1 + (c_2-c_1) r \x0c$

        where

        \x0c$ c_1 = x^e - d(1+\x07lpha \x08eta) \x0c$ 

        \x0c$ c_2 = x^e - d(1-\x07lpha \x08eta) \x0c$ 

        \x0c$ d = \x0crac{x^r - x^e}{2} \x0c$ 
 

        and

        \x0c$ \x07lpha = \x0c$ 1 if i < j & -1 if i > j 

        \x0c$ \x08eta = \x0crac{|j-i|-1}{b-2} \x0c$

        where b is the size of the population.

        Adapted from ideas in Egea, "An evolutionary method for complex-
        process optimization."

        Applies simple_bounds() to ensure all solutions lie within the design
        space by adapting the step size to the size of the design space.

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \\e list: A list of the identities of the chosen index for
            each child.
        """
        intDiscID = self.iID + self.dID
        varID = self.cID
        children = []
        used = []
        for i in range(0, int(len(pop) * self.fracElite), 1):
            j = int(rand() * len(pop))
            while j == i or j in used:
                j = int(rand() * len(pop))

            used.append(i)
            d = (pop[j] - pop[i]) / 2.0
            if i < j:
                alpha = 1
            else:
                alpha = -1
            beta = (abs(j - i) - 1) / (len(pop) - 2)
            c1 = pop[i] - d * (1 + alpha * beta)
            c2 = pop[i] + d * (1 - alpha * beta)
            tmp = c1 + (c2 - c1) * rand(len(pop[i]))
            tmp = tmp * varID + np.round(tmp * intDiscID)
            children.append(simple_bounds(tmp, self.lb, self.ub))

        return (
         children, used)

    def inversion_crossover(self, pop):
        """!
        Generate new designs by using inver-over on combinatorial variables.
        Adapted from ideas in Tao, "Iver-over Operator for the TSP."

        Although logic originally designed for combinatorial variables, it
        works for all variables and is used for all here.  The primary
        difference is the number of times that the crossover is performed.

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        """
        children, tmpNonComb, used = ([] for i in range(3))
        for i in range(0, int(len(pop) * self.fracElite), 1):
            r = int(rand() * len(pop))
            while r == i:
                r = int(rand() * len(pop))

            if sum(self.cID + self.dID + self.iID) != 0:
                nonComb1 = pop[i][:np.where(self.cID + self.dID + self.iID == 1)[0][(-1)] + 1]
                nonComb2 = pop[r][:np.where(self.cID + self.dID + self.iID == 1)[0][(-1)] + 1]
            if sum(self.xID) != 0:
                comb1 = pop[i][:np.where(self.xID == 1)[0][(-1)] + 1]
                comb2 = pop[r][:np.where(self.xID == 1)[0][(-1)] + 1]
            if sum(self.cID + self.dID + self.iID) != 0:
                c = int(rand() * len(nonComb1))
                if rand() > 0.5:
                    tmpNonComb.append(np.array(nonComb1[0:c + 1].tolist() + nonComb2[c + 1:].tolist()))
                else:
                    tmpNonComb.append(np.array(nonComb2[0:c + 1].tolist() + nonComb1[c + 1:].tolist()))
                used.append(i)
            if sum(self.xID) != 0:
                c = int(rand() * len(comb1))
                for c1 in range(c, len(comb1), 1):
                    d2 = (contains_sublist(comb2, comb1[c1]) + 1) % len(comb1)
                    d1 = contains_sublist(comb1, comb2[d2])
                    c2 = contains_sublist(comb2, comb1[((d1 + 1) % len(comb1))]) % len(comb1)
                    tmp1 = cp.copy(comb1)
                    if c1 < d1:
                        tmp1[(c1 + 1):(d1 + 1)] = list(reversed(tmp1[c1 + 1:d1 + 1]))
                    else:
                        tmp1[d1:c1] = list(reversed(tmp1[d1:c1]))
                    tmp2 = cp.copy(comb2)
                    if c2 < d2:
                        tmp2[c2:d2] = list(reversed(tmp2[c2:d2]))
                    else:
                        tmp2[(d2 + 1):(c2 + 1)] = list(reversed(tmp2[d2 + 1:c2 + 1]))
                    if sum(self.cID + self.dID + self.iID) == 0 and sum(self.xID) != 0:
                        children.append(tmp1)
                        children.append(tmp2)
                    elif sum(self.cID + self.dID + self.iID) != 0 and sum(self.xID) != 0:
                        children.append(np.concatenate(tmpNonComb[(-1)], tmp1))
                        children.append(np.concatenate(tmpNonComb[(-1)], tmp2))
                    used.append(i)
                    used.append(r)

        if sum(self.cID + self.dID + self.iID) != 0 and sum(self.xID) == 0:
            children = tmpNonComb
        return (
         children, used)

    def crossover(self, pop):
        r"""!
        Generate new children using distance based crossover strategies on
        the top parent. Ideas adapted from Walton "Modified Cuckoo Search: A
        New Gradient Free Optimisation Algorithm" and Storn "Differential
        Evolution - A Simple and Efficient Heuristic for Global Optimization
        over Continuous Spaces"

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """
        intDiscID = self.iID + self.dID
        varID = self.cID
        goldenRatio = (1.0 + sqrt(5)) / 2.0
        dx = np.zeros_like(pop[0])
        children = []
        used = []
        for i in range(0, int(self.fracElite * len(pop)), 1):
            r = int(rand() * self.population)
            while r in used or r == i:
                r = int(rand() * self.population)

            used.append(i)
            children.append(cp.deepcopy(pop[r]))
            dx = abs(pop[i] - children[i]) / goldenRatio
            children[i] = children[i] + dx * varID + np.round(dx * intDiscID)
            children[i] = simple_bounds(children[i], self.lb, self.ub)

        return (
         children, used)

    def mutate(self, pop):
        """!
        Generate new children by adding a weighted difference between two
        population vectors to a third vector.  Ideas adapted from Storn,
        "Differential Evolution - A Simple and Efficient Heuristic for Global
        Optimization over Continuous Spaces" and Yang, "Nature Inspired
        Optimization Algorithms"

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        """
        intDiscID = self.iID + self.dID
        varID = self.cID
        children = []
        k = rand(len(pop), len(pop[0])) > self.fracMutation * rand()
        childn1 = cp.copy(permutation(pop))
        childn2 = cp.copy(permutation(pop))
        r = rand()
        for j in range(0, len(pop), 1):
            n = np.array(childn1[j] - childn2[j])
            stepSize = r * n * varID + (n * intDiscID).astype(int)
            tmp = (pop[j] + stepSize * k[j, :]) * varID + (pop[j] + stepSize * k[j, :]) * intDiscID % (self.ub + 1 - self.lb)
            children.append(simple_bounds(tmp, self.lb, self.ub))

        return children

    def two_opt(self, pop):
        r"""!
        Generate new children using the two_opt operator.

        Ideas adapted from:
        Lin and Kernighan, "An Effective Heurisic Algorithm for the Traveling
        Salesman Problem"

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """
        children, used = ([] for i in range(2))
        for i in range(0, int(self.fracElite * len(pop)), 1):
            breaks = np.sort(rand(2) * len(pop[i]) // 1)
            breaks[1] = int(breaks[0] + tlf(1, 1)[(0, 0)] * len(pop[i])) % len(pop[i])
            while abs(breaks[0] - breaks[1]) < 2:
                breaks[1] = int(breaks[0] + tlf(1, 1)[(0, 0)] * len(pop[i])) % len(pop[i])
                np.sort(breaks)

            children.append(pop[i])
            children[(-1)][(int(breaks[0])):(int(breaks[1]))] = list(reversed(pop[i][int(breaks[0]):int(breaks[1])]))
            used.append(i)

        return (
         children, used)

    def three_opt(self, pop):
        r"""!
        Generate new children using the three_opt operator.

        Ideas adapted from:
        Lin and Kernighan, "An Effective Heurisic Algorithm for the Traveling
        Salesman Problem"

        @param self: <em> GnoweeHeuristic pointer </em> 

            The GnoweeHeuristics pointer. 

        @param pop: <em> list of arrays </em> 

            The current parent sets of design variables representing system
            designs for the population. 

        @return <em> list of arrays: </em>   The proposed children sets of
            design variables representing the updated design parameters.
        @return \e list: A list of the identities of the chosen index for
            each child.
        """
        children, used = ([] for i in range(2))
        for i in range(0, self.population, 1):
            tmp = []
            breaks = np.sort(rand(3) * len(pop[i]) // 1)
            while breaks[1] == breaks[0] or breaks[1] == breaks[2]:
                breaks[1] = rand() * len(pop[i]) // 1
                breaks = np.sort(breaks)

            while breaks[2] == breaks[0]:
                breaks[2] = rand() * len(pop[i]) // 1

            breaks = np.sort(breaks)
            tmp[0:(int(breaks[0]))] = pop[i][0:int(breaks[0])]
            tmp[(len(tmp)):(int(len(tmp) + breaks[2] - breaks[1]))] = pop[i][int(breaks[1]):int(breaks[2])]
            tmp[(len(tmp)):(int(len(tmp) + breaks[1] - breaks[0]))] = pop[i][int(breaks[0]):int(breaks[1])]
            tmp[(len(tmp)):(int(len(tmp) + breaks[2] - len(pop[i])))] = pop[i][int(breaks[2]):len(pop[i])]
            children.append(tmp)
            used.append(i)
            tmp = []
            tmp[0:(int(breaks[0]))] = pop[i][0:int(breaks[0])]
            tmp[(len(tmp)):(int(len(tmp) + breaks[1] - breaks[0]))] = list(reversed(pop[i][int(breaks[0]):int(breaks[1])]))
            tmp[(len(tmp)):(int(len(tmp) + breaks[2] - breaks[1]))] = reversed(pop[i][int(breaks[1]):int(breaks[2])])
            tmp[(len(tmp)):(int(len(tmp) + breaks[2] - len(pop[i])))] = pop[i][int(breaks[2]):len(pop[i])]
            children.append(tmp)
            used.append(i)

        return (
         children, used)

    def population_update_multi(self, parents, children, timeline=None, genUpdate=0, adoptedParents=[], mhFrac=0.0, randomParents=False):
        r"""!
        Calculate fitness, apply constraints, if present, and update the
        population if the children are better than their parents. Several
        optional inputs are available to modify this process. Refer to the
        input param documentation for more details.

        @param parents: <em> list of parent objects </em> 

            The current parents representing system designs. 

        @param children: <em> list of arrays </em> 

            The children design variables representing new system designs. 

        @param timeline: <em> list of history objects </em> 

            The histories of the optimization process containing best design,
            fitness, generation, and function evaluations. 

        @param genUpdate: \e integer 

            Indicator for how many generations to increment the counter by.
            Genenerally 0 or 1. 

        @param adoptedParents: \e list 

            A list of alternative parents to compare the children against.
            This alternative parents are then held accountable for not being
            better than the children of others. 

        @param mhFrac: \e float 

            The Metropolis-Hastings fraction.  A fraction of the otherwise
            discarded parents will be evaluated for acceptance against the
            greater population. 

        @param randomParents: \e boolean 

            If True, a random parent will be selected for comparison to the
            children. No one is safe. 

        @return <em> list of parent objects: </em> The current parents
            representing system designs. 

        @return \e integer:  The number of replacements made. 

        @return <em> list of history objects: </em> If an initial timeline was
            provided, retunrs an updated history of the optimization process
            containing best design, fitness, generation, and function
            evaluations.
        """
        numFunctions = self.numObjectiveFunctions
        if self.isFunctionList == 0:
            if not hasattr(self.objective.func, '__call__'):
                raise AssertionError('Invalid \tion handle.')
            assert self.dID != [] and np.sum(self.dID + self.xID) == len(self.discreteVals), ('A map must exist for each discrete variable. {} discrete variables, and {} maps provided.').format(np.sum(self.dID), len(self.discreteVals))
        if sum(self.dID) + sum(self.xID) != 0:
            for c in range(0, len(children)):
                children[c] = self.map_to_discretes(children[c])

            for p in parents:
                p.variables = self.map_to_discretes(p.variables)

        replace = 0
        numFunctions = self.numObjectiveFunctions
        if numFunctions == 1:
            for i in range(0, len(children), 1):
                fnew = self.objective.func(children[i])
                if fnew > self.penalty:
                    self.penalty = fnew

            feval = 0
            for i in range(0, len(children), 1):
                if randomParents:
                    j = int(rand() * len(parents))
                elif len(adoptedParents) == len(children):
                    j = adoptedParents[i]
                else:
                    j = i
                fnew = self.objective.func(children[i])
                for con in self.constraints:
                    fnew += con.func(children[i])

                feval += 1
                if fnew < parents[j].fitness:
                    parents[j].fitness = fnew
                    parents[j].variables = cp.copy(children[i])
                    parents[j].changeCount += 1
                    parents[j].stallCount = 0
                    replace += 1
                    if parents[j].changeCount >= 25 and j >= self.population * self.fracElite:
                        parents[j].variables = self.initialize(1, 'random').flatten()
                        parents[j].variables = self.map_to_discretes(parents[j].variables)
                        fnew = self.objective.func(parents[j].variables)
                        for con in self.constraints:
                            fnew += con.func(parents[j].variables)

                        parents[j].fitness = fnew
                        parents[j].changeCount = 0
                else:
                    parents[j].stallCount += 1
                    if parents[j].stallCount > 50000 and j != 0:
                        parents[j].variables = self.initialize(1, 'random').flatten()
                        parents[j].variables = self.map_to_discretes(parents[j].variables)
                        fnew = self.objective.func(parents[j].variables)
                        for con in self.constraints:
                            fnew += con.func(parents[j].variables)

                        parents[j].fitness = fnew
                        parents[j].changeCount = 0
                        parents[j].stallCount = 0
                    r = int(rand() * len(parents))
                    if r <= mhFrac:
                        r = int(rand() * len(parents))
                        if fnew < parents[r].fitness:
                            parents[r].fitness = fnew
                            parents[r].variables = cp.copy(children[i])
                            parents[r].changeCount += 1
                            parents[r].stallCount += 1
                            replace += 1

            parents.sort(key=lambda x: x.fitness)
        else:
            numVariables = len(self.varType)
            objectivelist = self.objective
            num_cont_int_bin_variables = len(self.lb)
            var_range = []
            for k in range(0, num_cont_int_bin_variables):
                var_range += (self.lb[k], self.ub[k])

            num_Features = num_cont_int_bin_variables
            problem = Problem(num_of_variables=num_Features, objectives=objectivelist, variables_range=var_range)
            num_adopted_parents = len(adoptedParents)
            num_parents = len(parents)
            num_children = len(children)
            populationSize = num_parents
            num_Features = num_cont_int_bin_variables
        if timeline != None:
            if len(timeline) < 2:
                timeline.append(Event(1, feval, parents[0].fitness, parents[0].variables))
            elif parents[0].fitness < timeline[(-1)].fitness and abs((timeline[(-1)].fitness - parents[0].fitness) / parents[0].fitness) > self.convTol:
                timeline.append(Event(timeline[(-1)].generation, timeline[(-1)].evaluations + feval, parents[0].fitness, parents[0].variables))
            else:
                timeline[(-1)].generation += genUpdate
                timeline[(-1)].evaluations += feval
        if sum(self.dID) + sum(self.xID) != 0:
            for p in parents:
                p.variables = self.map_from_discretes(p.variables)

        if timeline != None:
            return (parents, replace, timeline)
        else:
            return (
             parents, replace)
            return


def simple_bounds(child, lb, ub):
    r"""!
    @ingroup GnoweeHeuristics
    Application of problem boundaries to generated solutions. If outside of the
    boundaries, the variable defaults to the boundary.

    @param child: \e array 

        The proposed new system designs. 

    @param lb: \e array 

        The lower bounds of the design variable(s). 

    @param ub: \e array 

        The upper bounds of the design variable(s). 

    @return \e array: The new system design that is within problem
        boundaries. 

    """
    assert len(lb) == len(ub), 'Lower and upper bounds have different #s of design variables in simple_bounds function.'
    assert len(lb) == len(child), 'Bounds and child have different #s of design variables in simple_bounds function.'
    for i in range(0, len(child), 1):
        if child[i] < lb[i]:
            child[i] = lb[i]

    for i in range(0, len(child), 1):
        if child[i] > ub[i]:
            child[i] = ub[i]

    return child


def rejection_bounds(parent, child, stepSize, lb, ub):
    r"""!
    @ingroup GnoweeHeuristics
    Application of problem boundaries to generated solutions. Adjusts step size
    for all rejected solutions until within the boundaries.

    @param parent: \e array 

        The current system designs. 

    @param child: \e array 

        The proposed new system designs. 

    @param stepSize: \e float 

        The stepsize for the permutation. 

    @param lb: \e array 

        The lower bounds of the design variable(s). 

    @param ub: \e array 

        The upper bounds of the design variable(s). 

    @return \e array: The new system design that is within problem
        boundaries. 

    """
    assert len(lb) == len(ub), 'Lower and upper bounds have different #s of design variables in rejection_bounds function.'
    assert len(lb) == len(child), 'Bounds and child have different #s of design variables in rejection_bounds function.'
    for i in range(0, len(child), 1):
        stepReductionCount = 0
        while child[i] < lb[i] or child[i] > ub[i]:
            if stepReductionCount >= 5:
                child[i] = cp.copy(parent[i])
            else:
                stepSize[i] = stepSize[i] / 2.0
                child[i] = child[i] - stepSize[i]
                stepReductionCount += 1

    return child


def contains_sublist(lst, sublst):
    r"""!
    @ingroup GnoweeHeuristics
    Find index of sublist, if it exists.

    @param lst: \e list 

        The list in which to search for sublst. 

    @param sublst: \e list 

        The list to search for. 

    @return \e integer: Index location of sublst in lst. 

    """
    for i in range(0, len(lst), 1):
        if sublst == lst[i]:
            return i
