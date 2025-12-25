import numpy as np
import numpy.typing as npt


def roulette_wheel[T](population: npt.NDArray[T], fitnesses: npt.NDArray[np.float32], amount: int, replace: bool,
                      rng: np.random.Generator) -> tuple[npt.NDArray[T], npt.NDArray[np.float32]]:
    probabilities = fitnesses / fitnesses.sum()
    selected_indices = rng.choice(population.shape[0], size=amount, replace=replace, p=probabilities)
    return population[selected_indices], fitnesses[selected_indices]
