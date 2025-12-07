import numpy as np
import numpy.typing as npt


def n_point[T](parents: npt.NDArray[T], start: npt.NDArray[np.uintp] | None, pc: float,
               rng: np.random.Generator) -> npt.NDArray[T]:
    if start is None:
        start = rng.integers(parents.shape[1], size=parents.shape[0])[:, None]
    original_shape = parents.shape[2:]
    parents = parents.reshape((parents.shape[0], parents.shape[1], -1))

    indices = (((rng.random((parents.shape[0], parents.shape[2]), dtype=np.float32) < pc) + start).cumsum(axis=1) % parents.shape[1])
    return (parents[np.arange(parents.shape[0])[:, None], indices, np.arange(parents.shape[2])[None, :]]).reshape(parents.shape[0], *original_shape)
