""" Utility functions for scattering transforms.

These aren't useful for the transform, itself, but for dealing with outputs, misc. data handling, etc.
"""

from typing import List

import jax.numpy as jnp
from jaxtyping import Array


def group_fields_by_size(
    field1: List[Array],
    field2: List[List[Array]],
) -> List[Array]:
    """group_fields_by_size Group scattering fields from different layers by size.

    Args:
        field1 (List[Complex[Array]]): Scattering fields from 1st layer of transform.
        field2 (List[List[Complex[Array]]]): Scattering fields from 2nd layer of transform.

    Returns:
        List[Complex[Array]]: list of 6D arrays (BCLMHW) where elements correspond to the same spatial size, sorted in descending order.
    """
    shapes = list(map(lambda f: f.shape[-1], field1))
    idxs = [list() for _ in range(len(field2))]
    for j1 in range(len(field2)):
        for j2 in range(len(field2[j1])):
            shape = field2[j1][j2].shape[-1]
            idx = jnp.argwhere(jnp.asarray(shapes == shape))
            if len(idx):
                idxs[j1].append(int(idx[0][0]))
            else:
                idxs[j1].append(-1)
    for f2i, idxl in enumerate(idxs):
        for f2i2, idx in enumerate(idxl):
            if idx > 0:
                field1[idx] = jnp.concatenate(
                    [field1[idx][:, :, :, None, ...], field2[f2i][f2i2]], axis=3
                )
    return field1
