# uncompyle6 version 3.5.0
# Python bytecode 2.7 (62211)
# Decompiled from: Python 2.7.5 (default, Aug  7 2019, 00:51:29) 
# [GCC 4.8.5 20150623 (Red Hat 4.8.5-39)]
# Embedded file name: C:\Users\depila\Desktop\Graduate Research\GeneticAlgorithims\CreateNSGA2\NSGAOutline\ObjectiveFunction_Multi.py
# Compiled at: 2021-03-31 00:07:50
"""!
@file src/ObjectiveFunction.py
@package Gnowee

@defgroup ObjectiveFunction ObjectiveFunction

@brief Defines a class to perform objective function calculations.

This class contains the necessary functions and methods to create objective
functions and initialize the necessary parameters. The class is pre-stocked
with common benchmark functions for easy fishing.

Users can modify the this class to add additional functions following the
format of the functions currently in the class.

@author James Bevins

@date 23May17

@copyright <a href='../../licensing/COPYRIGHT'>&copy; 2017 UC
            Berkeley Copyright and Disclaimer Notice</a>
@license <a href='../../licensing/LICENSE'>GNU GPLv3.0+ </a>
"""
import numpy as np, operator
from math import sqrt, exp, log, cos, pi

class ObjectiveFunction_multi(object):
    """!
    @ingroup ObjectiveFunction
    This class creates a ObjectiveFunction object that can be used in
    optimization algorithms.
    """

    def __init__(self, method=None, objective=None):
        r"""!
        Constructor to build the ObjectiveFunction class.

        This class specifies the objective function to be used for a
        optimization process.

        @param self: <em> ObjectiveFunction pointer </em> 

            The ObjectiveFunction pointer. 

        @param method: \e string 

            The name of the objective function to evaluate. 

        @param objective: <em> integer, float, or numpy array </em> 

            The desired objective associated with the optimization.  The
            chosen value and type must be compatible with the optiization
            function chosen. This is used in objective functions that involve
            a comparison against a desired outcome. 

        """
        self._FUNC_DICT = {'spring': self.spring, 'mi_spring': self.mi_spring, 
           'welded_beam': self.welded_beam, 
           'pressure_vessel': self.pressure_vessel, 
           'mi_pressure_vessel': self.mi_pressure_vessel, 
           'speed_reducer': self.speed_reducer, 
           'mi_chemical_process': self.mi_chemical_process, 
           'ackley': self.ackley, 
           'shifted_ackley': self.shifted_ackley, 
           'dejong': self.dejong, 
           'shifted_dejong': self.shifted_dejong, 
           'easom': self.easom, 
           'shifted_easom': self.shifted_easom, 
           'griewank': self.griewank, 
           'shifted_griewank': self.shifted_griewank, 
           'rastrigin': self.rastrigin, 
           'shifted_rastrigin': self.shifted_rastrigin, 
           'rosenbrock': self.rosenbrock, 
           'shifted_rosenbrock': self.shifted_rosenbrock, 
           'tsp': self.tsp}
        if method != None and type(method) == str:
            self.set_obj_func(method)
        else:
            self.func = method
        self.objective = objective
        return

    def __repr__(self):
        r"""!
        ObjectiveFunction class param print function.

        @param self: \e ObjectiveFunction pointer 

            The ObjectiveFunction pointer. 

        """
        return ('ObjectiveFunction({}, {})').format(self.func.__name__, self.objective)

    def __str__(self):
        r"""!
        Human readable ObjectiveFunction print function.

        @param self: \e ObjectiveFunction pointer 

            The ObjectiveFunction pointer. 

        """
        header = [
         '  ObjectiveFunction:']
        header += [('Function: {}').format(self.func.__name__)]
        header += [('Objective: {}').format(self.objective)]
        return ('\n').join(header) + '\n'

    def set_obj_func(self, funcName):
        r"""!
        Converts an input string name for a function to a function handle.

        @param self: \e pointer 

            The ObjectiveFunction pointer. 

        @param funcName \e string 

             A string identifying the objective function to be used. 

        """
        if hasattr(funcName, '__call__'):
            self.func = funcName
        else:
            try:
                self.func = getattr(self, funcName)
                assert hasattr(self.func, '__call__'), 'Invalid function handle'
            except KeyError:
                print ('ERROR: The function specified does not exist in the ObjectiveFunction class or the _FUNC_DICT. Allowable methods are {}').format(self._FUNC_DICT)

    def spring(self, u):
        r"""!
        Spring objective function.

        Nearly optimal Example: 

        u = [0.05169046, 0.356750, 11.287126] 

        fitness = 0.0126653101469

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 3, 'Spring design needs to specify D, W, and L and only those 3 parameters.'
        assert u[0] != 0 and u[1] != 0 and u[2] != 0, ('Design values {} cannot be zero.').format(u)
        fitness = (2 + u[2]) * u[0] ** 2 * u[1]
        return fitness

    def mi_spring(self, u):
        r"""!
        Spring objective function.

        Optimal Example: 

        u = [1.22304104, 9, 36] = [1.22304104, 9, 0.307]

        fitness = 2.65856

        Taken from Lampinen, "Mixed Integer-Discrete-Continuous Optimization
        by Differential Evolution"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e float: The fitness associated with the specified input. 

        """
        assert len(u) == 3, 'Spring design needs to specify D, N, and d and only those 3 parameters.'
        D = u[0]
        N = u[1]
        d = u[2]
        Fmax = 1000
        S = 189000.0
        Fp = 300
        sigmapm = 6.0
        sigmaw = 1.25
        G = 11.5 * 1000000
        lmax = 14
        dmin = 0.2
        Dmax = 3.0
        K = G * d ** 4 / (8 * N * D ** 3)
        sigmap = Fp / K
        Cf = (4 * (D / d) - 1) / (4 * (D / d) - 4) + 0.615 * d / D
        lf = Fmax / K + 1.05 * (N + 2) * d
        fitness = np.pi ** 2 * D * d ** 2 * (N + 2) / 4
        return fitness

    def welded_beam(self, u):
        r"""!
        Welded Beam objective function.

        Nearly optimal Example: 

        u = [0.20572965, 3.47048857, 9.0366249, 0.20572965] 

        fitness = 1.7248525603892848

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 4, 'Welded Beam design needs to specify 4 parameters.'
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0, ('Designvalues {} cannot be zero').format(u)
        em = 6000.0 * (14 + u[1] / 2.0)
        r = sqrt(u[1] ** 2 / 4.0 + ((u[0] + u[2]) / 2.0) ** 2)
        j = 2.0 * (u[0] * u[1] * sqrt(2) * (u[1] ** 2 / 12.0 + ((u[0] + u[2]) / 2.0) ** 2))
        tau_p = 6000.0 / (sqrt(2) * u[0] * u[1])
        tau_dp = em * r / j
        tau = sqrt(tau_p ** 2 + 2.0 * tau_p * tau_dp * u[1] / (2.0 * r) + tau_dp ** 2)
        sigma = 504000.0 / (u[3] * u[2] ** 2)
        delta = 65856000.0 / (30 * 1000000 * u[3] * u[2] ** 2)
        pc = 4.013 * (30.0 * 1000000) * sqrt(u[2] ** 2 * u[3] ** 6 / 36.0) / 196.0 * (1.0 - u[2] * sqrt(30.0 * 1000000 / (4.0 * (12.0 * 1000000))) / 28.0)
        fitness = 1.10471 * u[0] ** 2 * u[1] + 0.04811 * u[2] * u[3] * (14.0 + u[1])
        return fitness

    def pressure_vessel(self, u):
        r"""!
        Pressure vessel objective function.

        Nearly optimal obtained using Gnowee: 

        u = [0.778169, 0.384649, 40.319619, 199.999998] 

        fitness = 5885.332800

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 4, 'Pressure vesseldesign needs to specify 4 parameters.'
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0, ('Designvalues {} cannot be zero').format(u)
        fitness = 0.6224 * u[0] * u[2] * u[3] + 1.7781 * u[1] * u[2] ** 2 + 3.1661 * u[0] ** 2 * u[3] + 19.84 * u[0] ** 2 * u[2]
        return fitness

    def mi_pressure_vessel(self, u):
        r"""!
        Mixed Integer Pressure vessel objective function.

        Nearly optimal example: 

        u = [58.2298, 44.0291, 17, 9] 

        fitness = 7203.24

        Optimal example obtained with Gnowee: 

        u = [38.819876, 221.985576, 0.750000, 0.375000] 

        fitness = 5855.893191

        Taken from: "Nonlinear Integer and Discrete Programming in Mechanical
        Design Optimization"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 4, 'MI Pressure vessel design needs to specify 4 parameters.'
        R = u[0]
        L = u[1]
        ts = u[2]
        th = u[3]
        fitness = 0.6224 * R * ts * L + 1.7781 * R ** 2 * th + 3.1611 * ts ** 2 * L + 19.8621 * R * ts ** 2
        return fitness

    def speed_reducer(self, u):
        r"""!
        Speed reducer objective function.

        Nearly optimal example: 

        u = [58.2298, 44.0291, 17, 9] 

        fitness = 2996.34784914

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 7, 'Speed reducer design needs to specify 7 parameters.'
        assert u[0] != 0 and u[1] != 0 and u[2] != 0 and u[3] != 0 and u[4] != 0 and u[5] != 0 and u[6] != 0, ('Design values cannot be zero {}.').format(u)
        fitness = 0.7854 * u[0] * u[1] ** 2 * (3.3333 * u[2] ** 2 + 14.9334 * u[2] - 43.0934) - 1.508 * u[0] * (u[5] ** 2 + u[6] ** 2) + 7.4777 * (u[5] ** 3 + u[6] ** 3) + 0.7854 * (u[3] * u[5] ** 2 + u[4] * u[6] ** 2)
        return fitness

    def mi_chemical_process(self, u):
        r"""!
        Chemical process design mixed integer problem.

        Optimal example: 

        u = [(0.2, 0.8, 1.907878, 1, 1, 0, 1] 

        fitness = 4.579582

        Taken from: "An Improved PSO Algorithm for Solving Non-convex
        NLP/MINLP Problems with Equality Constraints"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated.
            [x1, x2, x3, y1, y2, y3, y4] 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 7, 'Chemical process design needs to specify 7 parameters.'
        fitness = (u[3] - 1) ** 2 + (u[4] - 2) ** 2 + (u[5] - 1) ** 2 - log(u[6] + 1) + (u[0] - 1) ** 2 + (u[1] - 2) ** 2 + (u[2] - 3) ** 2
        return fitness

    def ackley(self, u):
        r"""!
        Ackley Function: Mulitmodal, n dimensional

        Optimal example: 

        u = [0, 0, 0, 0, ... n-1] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The Ackley Function must have a dimension greater than 1.'
        fitness = -20 * exp(-1.0 / 5.0 * sqrt(1.0 / len(u) * sum(u[i] ** 2 for i in range(len(u))))) - exp(1.0 / len(u) * sum(cos(2 * pi * u[i]) for i in range(len(u)))) + 20 + exp(1)
        return fitness

    def shifted_ackley(self, u):
        r"""!
        Ackley Function: Mulitmodal, n dimensional
        Ackley Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: 

        u = [0, 1, 2, 3, ... n-1] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The Shifted Ackley Function must have a dimension greater than 1.'
        fitness = -20 * exp(-1.0 / 5.0 * sqrt(1.0 / len(u) * sum((u[i] - i) ** 2 for i in range(len(u))))) - exp(1.0 / len(u) * sum(cos(2 * pi * (u[i] - i)) for i in range(len(u)))) + 20 + exp(1)
        return fitness

    def dejong(self, u):
        r"""!
        De Jong Function: Unimodal, n-dimensional

        Optimal example: 

        u = [0, 0, 0, 0, ... n-1] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The De Jong Function must have a dimension greater than 1.'
        fitness = sum(i ** 2 for i in u)
        return fitness

    def shifted_dejong(self, u):
        r"""!
        De Jong Function: Unimodal, n-dimensional
        De Jong Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: 

        u = [0, 1, 2, 3, ... n-1] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        Taken from: "Solving Engineering Optimization Problems with the
        Simple Constrained Particle Swarm Optimizer"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The Shifted De Jong Function must have a dimension greater than 1.'
        fitness = sum((u[i] - i) ** 2 for i in range(len(u)))
        return fitness

    def easom(self, u):
        r"""!
        Easom Function: Multimodal, n-dimensional

        Optimal example: 

        u = [pi, pi] 

        fitness = 1.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 2, 'The Easom Function must have a dimension of 2.'
        fitness = -cos(u[0]) * cos(u[1]) * exp(-(u[0] - pi) ** 2 - (u[1] - pi) ** 2)
        return fitness

    def shifted_easom(self, u):
        r"""!
        Easom Function: Multimodal, n-dimensional
        Easom Function that is shifted from the symmetric pi, pi optimimum.

        Optimal example: 

        u = [pi, pi+1] 

        fitness = 1.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) == 2, 'The Easom Function must have a dimension of 2.'
        fitness = -cos(u[0]) * cos(u[1] - 1) * exp(-(u[0] - pi) ** 2 - (u[1] - 1 - pi) ** 2)
        return fitness

    def griewank(self, u):
        r"""!
        Griewank Function: Multimodal, n-dimensional

        Optimal example: 

        u = [0, 0, 0, ..., 0] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1 and len(u) <= 600, 'The Shifted Griewank Function must have a dimension between 1 and 600.'
        fitness = 1.0 / 4000.0 * sum(u[i] ** 2 for i in range(len(u))) - prod(cos(u[i] / sqrt(i + 1)) for i in range(len(u))) + 1.0
        return fitness

    def shifted_griewank(self, u):
        r"""!
        Griewank Function: Multimodal, n-dimensional
        Griewank Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: 

        u = [0, 1, 2, ..., n-1] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

            The design parameters to be evaluated. 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1 and len(u) <= 600, 'The Shifted Griewank Function must have a dimension between 1 and 600.'
        fitness = 1.0 / 4000.0 * sum((u[i] - i) ** 2 for i in range(len(u))) - prod(cos((u[i] - i) / sqrt(i + 1)) for i in range(len(u))) + 1.0
        return fitness

    def rastrigin(self, u):
        r"""!
        Rastrigin Function: Multimodal, n-dimensional

        Optimal example: 

        u = [0, 0, 0, ..., 0] 

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The Rastrigin Function must have a dimension greater than 1.'
        fitness = 10.0 * len(u) + sum(u[i] ** 2 - 10.0 * np.cos(2.0 * np.pi * u[i]) for i in range(len(u)))
        return fitness

    def shifted_rastrigin(self, u):
        r"""!
        Rastrigin Function: Multimodal, n-dimensional
        Rastrigin Function that is shifted from the symmetric 0, 0, 0, ..., 0
        optimimum.

        Optimal example: 

        u = [0, 1, 2, ..., n-1] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The Shifted Rastrigin Function must have a dimension greater than 1.'
        fitness = 10.0 * len(u) + sum((u[i] - i) ** 2 - 10.0 * np.cos(2.0 * np.pi * (u[i] - i)) for i in range(len(u)))
        return fitness

    def rosenbrock(self, u):
        r"""!
        Rosenbrock Function: uni-modal, n-dimensional.

        Optimal example: 

        u = [1, 1, 1, ..., 1] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The Rosenbrock Function must have a dimension greater than 1.'
        fitness = sum((u[i] - 1) ** 2 + 100.0 * (u[(i + 1)] - u[i] ** 2) ** 2 for i in range(len(u) - 1))
        return fitness

    def shifted_rosenbrock(self, u):
        r"""!
        Rosenbrock Function: uni-modal, n-dimensional
        Rosenbrock Function that is shifted from the symmetric 0,0,0...0
        optimimum.

        Optimal example: 

        u = [1, 2, 3, ...n] 

        fitness = 0.0

        Taken from: "Nature-Inspired Optimization Algorithms"

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        assert len(u) >= 1, 'The Shifted Rosenbrock Function must have a dimension greater than 1.'
        fitness = sum((u[i] - 1 - i) ** 2 + 100.0 * (u[(i + 1)] - (i + 1) - (u[i] - i) ** 2) ** 2 for i in range(len(u) - 1))
        return fitness

    def tsp(self, u):
        r"""!
        Generic objective funtion to evaluate the TSP optimization by
        calculating total distance traveled.

        @param self: <em> pointer </em> 

            The ObjectiveFunction pointer. 

        @param u: \e array 

        @return \e array: The fitness associated with the specified input. 

        @return \e array: The assessed value for each constraint for the
            specified input. 

        """
        fitness = 0
        for i in range(1, len(u), 1):
            fitness = fitness + round(sqrt((u[i][0] - u[(i - 1)][0]) ** 2 + (u[i][1] - u[(i - 1)][1]) ** 2))

        fitness = fitness + round(sqrt((u[0][0] - u[(-1)][0]) ** 2 + (u[0][1] - u[(-1)][1]) ** 2))
        return fitness


def prod(iterable):
    r"""!
    @ingroup ObjectiveFunction
    Computes the product of a set of numbers (ie big PI, mulitplicative
    equivalent to sum).

    @param iterable: <em> list or array or generator </em>
        Iterable set to multiply.

    @return \e float: The product of all of the items in iterable
    """
    return reduce(operator.mul, iterable, 1)
