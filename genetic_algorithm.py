from collections.abc import Callable, Generator

import numpy as np
import numpy.typing as npt


def evolve[T](population: npt.NDArray[T],
              fitness: Callable[[npt.NDArray[T]], npt.NDArray[np.float32]],
              parent_selector: Callable[[npt.NDArray[T], npt.NDArray[np.float32], int], tuple[
                  npt.NDArray[T], npt.NDArray[np.float32]]],
              offsprings_per_generation: int,
              parents_per_offspring: int,
              offspring_selector: Callable[[npt.NDArray[T], npt.NDArray[np.float32], int], tuple[
                  npt.NDArray[T], npt.NDArray[np.float32]]],
              reproduce: Callable[[npt.NDArray[T], npt.NDArray[np.uintp] | None, float], npt.NDArray[T]],
              first_parent_offset: Callable[[int], npt.NDArray[np.uintp] | None],
              pc: float,
              mutate: Callable[[npt.NDArray[T], float, Callable[[int], npt.NDArray[T]]], None],
              mutation_generator: Callable[[int], npt.NDArray[T]],
              pm: float,
              stop: Callable[[npt.NDArray[T], npt.NDArray[np.float32], int], bool]
              ) -> Generator[tuple[npt.NDArray[T], npt.NDArray[np.float32]]]:
    original_shape = population.shape
    population = population.reshape(original_shape[0], -1)
    fitnesses = fitness(population)
    generation = 0
    while not stop(population, fitnesses, generation):
        parents = (parent_selector(population, fitnesses, offsprings_per_generation * parents_per_offspring)[0]
                   .reshape((offsprings_per_generation, parents_per_offspring, -1)))
        offsprings = reproduce(parents, first_parent_offset(generation), pc)
        mutate(offsprings, pm, lambda n: mutation_generator(n))
        population = np.concatenate((population, offsprings), axis=0)
        fitnesses: npt.NDArray = fitness(population)
        population, fitnesses = offspring_selector(population, fitnesses, original_shape[0])
        generation += 1
        yield population.reshape(original_shape), fitnesses
