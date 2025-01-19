""" Utility functions for scattering transforms.

These aren't useful for the transform, itself, but for dealing with outputs, misc. data handling, etc.
"""

import math
from typing import List

import jax
import jax.numpy as jnp
import numpy
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Num


def apply_filter_bank(
    num_spatial_dims: int,
    x: Array,  # [channels x (spatial dimensions)]
    psi: Array,  # [filter dimensions x (spatial dimensions)]
    adicity: int,
    scale_subsample: int,
    scale_periodize: int | None = None,
) -> Array:
    """apply_filter_bank apply a bank of filters to all of the channels in the input.

    Args:
        num_spatial_dims (int): number of spatial dimensions in the input array
        x (Array): input array
        scale_subsample (int): scale to subsample the input by
        scale_periodize (int | None, optional): scale to periodize the filters by. Defaults to None. If None, will be the same as `scale_subsample`.

    Returns:
        Array: output will be shaped [channels x (filter dimensions) x (spatial dimensions)]
    """
    if scale_periodize is None:
        scale_periodize = scale_subsample
    if len(psi.shape) == num_spatial_dims:
        psi = jnp.expand_dims(psi, 0)
    # periodize filters appropriately, then flatten so that output goes from
    # [(filter variant dims) x (spatial dims)] -> [(concat'd filter variants) x (spatial dims)]
    filt_dim_shapes = list(psi.shape[:-num_spatial_dims])
    filt = flatten_filter_tensor(
        num_spatial_dims,
        periodize_filter(num_spatial_dims, psi, adicity, scale_periodize),
    )
    # subsample the field, as appropriate, shape is still [channels x (spatial dims)]
    field = subsample_field(num_spatial_dims, x, adicity, scale_subsample)
    out = jnp.expand_dims(filt, 0) * jnp.expand_dims(field, 1)
    out_space_dims = list(out.shape[-num_spatial_dims:])
    out_shape = (
        list(x.shape[:-num_spatial_dims]) + filt_dim_shapes + out_space_dims
    )
    return jnp.reshape(out, shape=out_shape)


def subsample_field(
    num_spatial_dims: int,
    x: Array,
    adicity: int,
    j: int,
) -> Array:
    """subsample_field subsample an input field by local averaging.

    Args:
        num_spatial_dims (int): number of spatial dimensions in input.
        x (Array): input array
        adicity (int): adicity of scale separation
        j (int): scale to subsample at (actual will be adicity**j)

    Returns:
        Array: subsampled input
    """
    if j == 0:
        return x
    k = adicity**j
    if len(x.shape) == num_spatial_dims:
        reshape_dims = list(range(num_spatial_dims))
        leading_shapes = list()
        mean_dims = list(range(0, num_spatial_dims + 1, 2))
    else:
        reshape_dims = [-i for i in reversed(range(1, num_spatial_dims + 1))]
        leading_shapes = list(x.shape[:-num_spatial_dims])
        mean_dims = [r * 2 for r in reshape_dims]
    subsamp_shapes = [
        x for xs in [[k, x.shape[i] // k] for i in reshape_dims] for x in xs
    ]
    reshape = leading_shapes + subsamp_shapes
    return jnp.mean(jnp.reshape(x, reshape), axis=tuple(mean_dims))


def periodize_filter(
    num_spatial_dims: int,
    x: Array,
    adicity: int,
    j: int,
) -> Array:
    """periodize_filter periodize the input fourier-domain input filter.

    Args:
        num_spatial_dims (int): number of spatial dimensions in input.
        x (Array): input array
        adicity (int): adicity of scale separation
        j (int): scale to periodize to (actual will be adicity**j)

    Raises:
        ValueError: if spatial dimensions <1 or >3

    Returns:
        Array: periodized filter
    """
    if num_spatial_dims < 1 or num_spatial_dims > 3:
        raise ValueError("invalid number of spatial dims, must be (1,2,3)")
    if len(x.shape) == num_spatial_dims:
        if num_spatial_dims == 1:
            return _periodize1d(x, adicity, j)
        elif num_spatial_dims == 2:
            return _periodize2d(x, adicity, j)
        else:
            return _periodize3d(x, adicity, j)
    else:  # need to map over other dimensions
        k = adicity**j
        inp_shape = x.shape
        rest = x.shape[:-num_spatial_dims]
        out_shape = list(inp_shape)
        for i in range(1, num_spatial_dims + 1):
            out_shape[-i] = out_shape[-i] // k
        reshape_flatchan = [math.prod(rest)] + [
            x.shape[-i] for i in reversed(range(1, num_spatial_dims + 1))
        ]
        if num_spatial_dims == 1:
            fun = Partial(_periodize1d, adicity=adicity, j=j)
        elif num_spatial_dims == 2:
            fun = Partial(_periodize2d, adicity=adicity, j=j)
        else:  # num_spatial_dims == 3
            fun = Partial(_periodize3d, adicity=adicity, j=j)
        out = jnp.reshape(
            jax.vmap(fun, 0)(jnp.reshape(x, reshape_flatchan)), out_shape
        )
        return out


def _periodize1d(
    x: Num[Array, " h"], adicity: int, j: int
) -> Num[Array, " {h//adicity**j}"]:
    k = adicity**j
    sze = x.shape[0]
    mask = numpy.ones(x.shape, numpy.float32)
    len_x = math.floor(sze * (1 - adicity ** (-j)))
    start_x = math.floor(sze * adicity ** (-j - 1))
    mask = mask.at[start_x : start_x + len_x].set(0)
    return jax.lax.conv_general_dilated(
        jnp.expand_dims(jnp.multiply(x, mask), (0, 1)),
        jnp.expand_dims(jnp.ones(k), (0, 1)),
        window_strides=1,
        padding="valid",
        rhs_dilation=(sze // k),
    )[0, 0, ...]


def _periodize2d(
    x: Num[Array, "h w"], adicity: int, j: int
) -> Num[Array, "{h//adicity**j} {w//adicity**j}"]:
    k = adicity**j
    hgt, wid = x.shape[0], x.shape[1]
    len_x = math.floor(hgt * (1 - adicity ** (-j)))
    start_x = math.floor(hgt * adicity ** (-j - 1))
    len_y = math.floor(wid * (1 - adicity ** (-j)))
    start_y = math.floor(wid * adicity ** (-j - 1))
    # mask out corners, as calculated, above
    mask = jnp.ones(x.shape, jnp.float32)
    mask = mask.at[start_x : start_x + len_x, :].set(0)
    mask = mask.at[:, start_y : start_y + len_y].set(0)
    #
    return jax.lax.conv_general_dilated(
        jnp.expand_dims(jnp.multiply(x, mask), (0, 1)),
        jnp.expand_dims(jnp.ones((k, k)), (0, 1)),
        window_strides=(1, 1),
        padding="valid",
        rhs_dilation=(hgt // k, wid // k),
    )[0, 0, ...]


def _periodize3d(
    x: Num[Array, "d h w"], adicity: int, j: int
) -> Num[Array, "{d//adicity**j} {h//adicity**j} {w//adicity**j}"]:
    k = adicity**j
    dep, hgt, wid = x.shape[0], x.shape[1], x.shape[2]
    len_z = math.floor(dep * (1 - adicity ** (-j)))
    start_z = math.floor(dep * adicity ** (-j - 1))
    len_x = math.floor(hgt * (1 - adicity ** (-j)))
    start_x = math.floor(hgt * adicity ** (-j - 1))
    len_y = math.floor(wid * (1 - adicity ** (-j)))
    start_y = math.floor(wid * adicity ** (-j - 1))
    mask = jnp.ones(x.shape, numpy.float32)
    mask = mask.at[start_z : start_z + len_z, ...].set(0)
    mask = mask.at[:, start_x : start_x + len_x, :].set(0)
    mask = mask.at[..., start_y : start_y + len_y].set(0)

    return jax.lax.conv_general_dilated(
        jnp.expand_dims(jnp.multiply(x, mask), (0, 1)),
        jnp.expand_dims(jnp.ones((k, k, k)), (0, 1)),
        window_strides=(1, 1, 1),
        padding="valid",
        rhs_dilation=(dep // k, hgt // k, wid // k),
    )[0, 0, ...]


def flatten_filter_tensor(
    num_spatial_dims: int,
    psi: Array,
) -> Array:
    """flatten_filter_tensor flatten filter parameter channels (along with possible image channels) into a single array axis.

    Args:
        num_spatial_dims (int): number of spatial dimensions in filter
        psi (Array): filter

    Returns:
        Array
    """
    reshape = [psi.shape[-i] for i in reversed(range(1, num_spatial_dims + 1))]
    reshape.insert(0, -1)
    return jnp.reshape(psi, reshape)


def group_fields_by_size(
    num_spatial_dims: int,
    field1: List[Array],
    field2: List[List[Array]],
) -> List[Array]:
    """group_fields_by_size Group scattering fields from different layers by size.

    Args:
        field1 (List[Array]): Scattering fields from 1st layer of transform.
        field2 (List[List[Array]]): Scattering fields from 2nd layer of transform.

    Returns:
        List[Array]: list of 6D arrays (CLMHW) where elements correspond to the same spatial size, sorted in descending order.
    """
    num_filt_dims = len(field1.shape[0]) - (num_spatial_dims + 1)
    field1 = [
        jnp.expand_dims(f, axis=range(num_filt_dims + 1, num_filt_dims + 3))
        for f in field1
    ]
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
                    [field1[idx], field2[f2i][f2i2]], axis=3
                )
    return field1
