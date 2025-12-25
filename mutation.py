from collections.abc import Callable

import numpy as np
import numpy.typing as npt


def redetermination[T](population: npt.NDArray[T], pm: float, rng: np.random.Generator,
                       value_gen: Callable[[int], npt.NDArray[T]]) -> None:
    mask = rng.random(population.shape, dtype=np.float32) < pm
    population[mask] = value_gen(mask.sum())
