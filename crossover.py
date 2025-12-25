import numpy as np
import numpy.typing as npt


def n_point[T](parents: npt.NDArray[T], start: npt.NDArray[np.uintp] | None, pc: float,
               rng: np.random.Generator) -> npt.NDArray[T]:
    if start is None:
        start = rng.integers(parents.shape[1], size=parents.shape[0], dtype=np.uint8)
    original_shape = parents.shape[2:]
    parents = parents.reshape((parents.shape[0], parents.shape[1], -1))

    # Calculate the parent for every cell of every offspring.
    # Each cell has a probability pc of switching to the next parent.
    # The start array indicates the starting parent for each offspring.
    # Cumulative sum used to turn the array into segments of parents.
    switches = (rng.random((parents.shape[0], parents.shape[2]), dtype=np.float32) < pc).astype(np.uint8)
    switches[:, 0] += start
    indices = (switches.cumsum(axis=1) % parents.shape[1])
    return (parents[np.arange(parents.shape[0])[:, None], indices, np.arange(parents.shape[2])[None, :]]).reshape(parents.shape[0], *original_shape)
