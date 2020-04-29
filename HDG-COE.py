from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import ExponentialRankingSelection, UniformCrossover, FlipBitMutation
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

import math as ma
import numpy as np


def M1(args):
    s = 0
    for i in args:
        s += 1 / i ** 2
    if s <= ((-b + ma.sqrt(b ** 2 + 8 * a * epsilon)) ** 2) / (2 * a):
        return 0
    else:
        return 10 ** 50


def _get_harmonic_num(order, power=1.0):
    if order == 1:
        value = 1.0
    else:
        value = 1.0 / (order**power) + \
            _get_harmonic_num(order - 1, power=power)
    return value


n = 0   # dimensionality
eta = 0  # supermum of M
s2 = 0
epsilon = 0

harmR1 = _get_harmonic_num(n, 1)
harmR12 = _get_harmonic_num(n, 0.5)
a = (harmR1 + harmR12) * (eta ** 2) + 2 * harmR1 * eta * s2
zeta = ma.sqrt(2 * ma.sqrt(-1 * n * ma.log(1 / n)) - 2 * ma.log(1 / n) + 1 * n)
b = 2 * ((1 * n) ** 0.25) * zeta * harmR1 * s2


sigma_ranges = [(0.1, 5)] * n
indv_template = BinaryIndividual(ranges=sigma_ranges, eps=0.01)
population = Population(indv_template=indv_template, size=50)
population.init()  # Initialize population with individuals.

# Use built-in operators here.
selection = ExponentialRankingSelection()
crossover = UniformCrossover(pc=0.5, pe=0.5)
mutation = FlipBitMutation(pm=0.5)

engine = GAEngine(population=population, selection=selection,
                  crossover=crossover, mutation=mutation)


@engine.fitness_register
@engine.minimize
def fitness(indv):
    ObjectF = 1
    Constraint = 0
    for i in indv.solution:
        ObjectF *= i
        Constraint += 1 / i ** 2
    return ObjectF + M1(indv.solution) * (Constraint -
                                          ((-b + ma.sqrt(b ** 2 + 8 * a * epsilon)) ** 2) / (2 * a))


def Generate_noise(sigma):
    mu_vector = np.zeros([1, n])
    covariance = np.diag(sigma)
    return np.random.multivariate_normal(mu_vector, covariance)


if '__main__' == __name__:
    engine.run(ng=150)
