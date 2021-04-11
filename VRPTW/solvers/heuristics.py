from VRPTW.structure import Problem, Route
import itertools

import copy
from functools import cmp_to_key
from typing import TypeVar, List

from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.config import store
from jmetal.core.operator import Mutation, Crossover, Selection
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.archive import BoundedArchive
from jmetal.util.density_estimator import CrowdingDistance, DensityEstimator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.neighborhood import Neighborhood
from jmetal.util.ranking import FastNonDominatedRanking, Ranking
from jmetal.util.comparator import Comparator, MultiComparator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')

class DummyHeuristic:
    def __init__(self, problem: Problem):
        self.problem: Problem = problem

    def get_solution(self):
        """Solution sampled from customer list, sorted by demand"""

        def get_available_customers():
            return sorted(filter(lambda x: not x.is_serviced, self.problem.customers), key=lambda x: x.due_date)

        solution = []
        while len(get_available_customers()) > 0:
            customers = get_available_customers()
            route = []
            for customer in customers:
                if Route(self.problem, route + [customer]).is_feasible:
                    customer.is_serviced = True
                    route.append(customer)
            solution.append(Route(self.problem, route))
        return solution


def two_opt(a, i, j):
    if i == 0:
        return a[j:i:-1] + [a[i]] + a[j + 1:]
    return a[:i] + a[j:i - 1:-1] + a[j + 1:]


def cross(a, b, i, j):
    return a[:i] + b[j:], b[:j] + a[i:]


def insertion(a, b, i, j):
    # print(a, b, i, j)
    if len(a) == 0:
        return a, b
    while i >= len(a):
        i -= len(a)
    return a[:i] + a[i + 1:], b[:j] + [a[i]] + b[j:]


def swap(a, b, i, j):
    # print(a, b, i, j)
    if i >= len(a) or j >= len(b):
        return a, b
    a, b = a.copy(), b.copy()
    a[i], b[j] = b[j], a[i]
    return a, b


class LocalSearch:
    def __init__(self, problem: Problem):
        self.problem: Problem = problem

    def optimize(self, solution: list) -> list:
        new_solution = list(solution)
        for i in range(len(new_solution)):
            is_stucked = False
            while not is_stucked:
                route = new_solution[i]
                is_stucked = True
                for k, j in itertools.combinations(range(len(route.customers)), 2):
                    new_route = Route(self.problem, two_opt(route.customers, k, j))
                    if new_route.is_feasible:
                        if new_route.total_distance < route.total_distance:
                            new_solution[i] = new_route
                            is_stucked = False
        return new_solution


class IteratedLocalSearch(LocalSearch):
    def __init__(self, problem: Problem, obj_func=None):
        super().__init__(problem)
        if not obj_func:
            obj_func = self.problem.obj_func
        self.obj_func = obj_func
        self.initial_solution = DummyHeuristic(problem).get_solution()

    def perturbation(self, routes: list) -> list:
        best = [Route(self.problem, route.customers) for route in routes]
        is_stucked = False
        while not is_stucked:
            is_stucked = True
            # Для всех возможных пар маршрутов
            for i, j in itertools.combinations(range(len(best)), 2):
                # Для всех возможных индексов в двух маршрутах
                for k, l in itertools.product(range(len(best[i].customers) + 2), range(len(best[j].customers) + 2)):
                    for func in [cross, insertion, swap]:
                        c1, c2 = func(best[i].customers, best[j].customers, k, l)
                        r1, r2 = Route(self.problem, c1), Route(self.problem, c2)
                        if r1.is_feasible and r2.is_feasible:
                            if r1.total_distance + r2.total_distance < best[i].total_distance + best[j].total_distance:
                                best[i] = r1
                                best[j] = r2
                                is_stucked = False
            best = list(filter(lambda x: len(x.customers) != 0, best))
        return best

    def execute(self):
        best = self.optimize(self.initial_solution)
        print(self.problem.print_canonical(best))
        print("Total distance", self.obj_func(best))

        is_stucked = False
        while not is_stucked:
            is_stucked = True
            new_solution = self.perturbation(best)
            new_solution = self.optimize(new_solution)
            if self.obj_func(new_solution) < self.obj_func(best):
                is_stucked = False
                best = list(filter(lambda x: len(x.customers) != 0, new_solution))
                print(self.problem.print_canonical(best))
                print("Total distance", self.obj_func(best))
        return best


class GuidedLocalSearch(IteratedLocalSearch):
    def __init__(self, problem: Problem, l=0.5):
        super().__init__(problem, self.augmented_obj_func)
        self.l = l
        self.initial_solution = DummyHeuristic(problem).get_solution()
        self.penalties = [[0 for _ in self.problem.customers] for _ in self.problem.customers]

    @staticmethod
    def numeric_edges(routes):
        return [edge for route in routes
                for edge in list(map(lambda x: (x[0].number, x[1].number), route.edges))]

    def augmented_obj_func(self, routes):
        g = self.problem.obj_func(routes)
        penalty_sum = 0
        for edge in self.numeric_edges(routes):
            penalty_sum = self.penalties[edge[0]][edge[1]]
        return g + self.l * penalty_sum

    def update_penalties(self, routes):
        util = [[0 for _ in self.problem.customers] for _ in self.problem.customers]
        for e in [e for route in map(lambda x: x.edges, routes) for e in route]:
            util[e[0].number][e[1].number] = \
                (e[0].distance(e[1]) / (1 + self.penalties[e[0].number][e[1].number]))
        max_util_value = max(max(x) for x in util)
        for i, _ in enumerate(util):
            for j, _ in enumerate(util[i]):
                if util[i][j] == max_util_value:
                    self.penalties[i][j] = self.penalties[i][j] + 1

    def execute(self):
        best = self.optimize(self.initial_solution)
        solutions = []
        k = 0
        while k < 20:
            is_stucked = True
            local_min = self.execute()
            self.update_penalties(local_min)
            solutions.append(local_min)
            k += 1
        return min(solutions, key=lambda x: self.problem.obj_func(x))


class MOCell(GeneticAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 neighborhood: Neighborhood,
                 archive: BoundedArchive,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([FastNonDominatedRanking.get_comparator(),
                                      CrowdingDistance.get_comparator()])),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator):
        """
        MOCEll implementation as described in:

        :param problem: The problem to solve.
        :param population_size: Size of the population.
        :param mutation: Mutation operator (see :py:mod:`jmetal.operator.mutation`).
        :param crossover: Crossover operator (see :py:mod:`jmetal.operator.crossover`).
        :param selection: Selection operator (see :py:mod:`jmetal.operator.selection`).
        """
        super().__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=1,
            mutation=mutation,
            crossover=crossover,
            selection=selection,
            termination_criterion=termination_criterion,
            population_evaluator=population_evaluator,
            population_generator=population_generator
        )
        self.dominance_comparator = dominance_comparator
        self.neighborhood = neighborhood
        self.archive = archive
        self.current_individual = 0
        self.current_neighbors = []
        self.solutions = []

        self.comparator = MultiComparator([FastNonDominatedRanking.get_comparator(),
                                           CrowdingDistance.get_comparator()])

    def init_progress(self) -> None:
            super().init_progress()
            for solution in self.solutions:
                self.archive.add(copy.copy(solution))


    def update_progress(self) -> None:
            super().update_progress()
            self.current_individual = (self.current_individual + 1) % self.population_size


    def selection(self, population: List[S]):
            parents = []

            self.current_neighbors = self.neighborhood.get_neighbors(self.current_individual, population)
            self.current_neighbors.append(self.solutions[self.current_individual])

            parents.append(self.selection_operator.execute(self.current_neighbors))
            if len(self.archive.solution_list) > 0:
                parents.append(self.selection_operator.execute(self.archive.solution_list))
            else:
                parents.append(self.selection_operator.execute(self.current_neighbors))

            return parents


    def reproduction(self, mating_population: List[S]) -> List[S]:
            number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

            if len(mating_population) % number_of_parents_to_combine != 0:
                raise Exception('Wrong number of parents')

            offspring_population = self.crossover_operator.execute(mating_population)
            self.mutation_operator.execute(offspring_population[0])

            return [offspring_population[0]]


    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
            result = self.dominance_comparator.compare(population[self.current_individual], offspring_population[0])

            if result == 1:  # the offspring individual dominates the current one
                population[self.current_individual] = offspring_population[0]
                self.archive.add(offspring_population[0])
            elif result == 0:  # the offspring and current individuals are non-dominated
                new_individual = offspring_population[0]

                self.current_neighbors.append(new_individual)

                ranking: Ranking = FastNonDominatedRanking()
                ranking.compute_ranking(self.current_neighbors)

                density_estimator: DensityEstimator = CrowdingDistance()
                for i in range(ranking.get_number_of_subfronts()):
                    density_estimator.compute_density_estimator(ranking.get_subfront(i))

                self.current_neighbors.sort(key=cmp_to_key(self.comparator.compare))
                worst_solution = self.current_neighbors[-1]

                self.archive.add(new_individual)
                if worst_solution != new_individual:
                    population[self.current_individual] = new_individual

            return population

    def get_result(self) -> R:
            return self.archive.solution_list


    def get_name(self) -> str:
            return 'MOCell'
