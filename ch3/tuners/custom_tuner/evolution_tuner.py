import random
import numpy as np
from nni.tuner import Tuner
from nni.utils import (
    OptimizeMode, extract_scalar_reward,
    json2space, json2parameter,
)


class Individual:
    """
    Individual: Single element of Population
    """

    def __init__(self, x, y, param_id = None) -> None:
        """
        x - coordinate
        y - coordinate
        param_id - NNI id of the parameter

        """
        self.param_id = param_id
        self.x = x
        self.y = y
        self.result = None

    def to_dict(self):
        return {'x': self.x, 'y': self.y}


class Population:

    def __init__(self) -> None:
        self.individuals = []

    def add(self, ind):
        self.individuals.append(ind)

    def get_by_param_id(self, param_id):
        """
        Returns Individual by param_id
        """
        for ind in self.individuals:
            if ind.param_id == param_id:
                return ind
        return None

    def get_first_virgin(self):
        for ind in self.individuals:
            if ind.param_id is None:
                return ind
        return None

    def get_population_with_result(self):
        """
        Returns Individuals that already has received their trial results
        """
        population_with_result = [ind for ind in self.individuals if ind.result is not None]
        return population_with_result

    def get_best_individual(self):
        """
        Returns the best Individual
        """
        sorted_population = sorted(self.get_population_with_result(), key = lambda ind: ind.result)
        return sorted_population[-1]

    def replace_worst(self, param_id):
        """
        Replaces the worst Individual by Best Individual mutant
        """
        population_with_result = self.get_population_with_result()
        sorted_population = sorted(population_with_result, key = lambda ind: ind.result)
        worst = sorted_population[0]
        self.individuals.remove(worst)
        best = self.get_best_individual()
        x = round(best.x + random.gauss(0, 1), 2)
        y = round(best.y + random.gauss(0, 1), 2)
        mutant = Individual(x, y, param_id)
        self.individuals.append(mutant)
        return mutant


class NewEvolutionTuner(Tuner):

    def __init__(self, optimize_mode = "maximize", population_size = 16) -> None:
        # Optimization Mode: 'maximize' | 'minimize'
        self.optimize_mode = OptimizeMode(optimize_mode)

        # Population Size
        self.population_size = population_size

        # Search Space JSON definition
        self.search_space_json = None

        # Tuner's Random State
        self.random_state = None

        # Population
        self.population = Population()

        # Search Space
        self.space = None

    def update_search_space(self, search_space):
        """
        Experiment Startup
        """
        self.search_space_json = search_space
        self.space = json2space(self.search_space_json)
        self.random_state = np.random.RandomState()

        # Population of Random Individuals is generated
        is_rand = dict()
        for item in self.space:
            is_rand[item] = True

        for _ in range(self.population_size):
            params = json2parameter(self.search_space_json, is_rand, self.random_state)
            ind = Individual(params['x'], params['y'])
            self.population.add(ind)

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Initially, we have a set of individuals that were generated at startup
        and were not passed to the Tuner (virgins).
        We take virgins and pass them to Tuner one by one.
        When no virgins are left, we are generating new individuals.
        """
        virgin = self.population.get_first_virgin()
        if virgin:
            virgin.param_id = parameter_id
            return virgin.to_dict()
        else:
            mutant = self.population.replace_worst(parameter_id)
            return mutant.to_dict()

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """
        Passing results to individuals
        """
        reward = extract_scalar_reward(value)
        ind = self.population.get_by_param_id(parameter_id)
        ind.result = reward
