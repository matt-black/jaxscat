""" Fourier-space 2D Scattering Transform.

Scattering transform for 2
"""

import math
from functools import partial
from typing import Tuple

import jax.numpy as jnp
import numpy
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Real


def scattering_coeffs(
    x: Complex[Array, "b c h w"],
    adicity: int,
    n_scales: int,
    phi: Complex[Array, "h w"],
    psi1: Complex[Array, "{n_scales} l h w"],
    psi2: Complex[Array, "{n_scales} l h w"] | None = None,
    strategy: str = "breadth",
) -> Tuple[
    Complex[Array, "b c"],
    Complex[Array, "b c j l"],
    Complex[Array, "b c j l l"],
]:
    """scattering_coeffs Compute scattering coefficients for input field, `x`.

    Args:
        x (Complex[Array, "batch channel height width"]): 4D input tensor. Coefficients are computed on last 2 dimensions.
        adicity (int): logarithm-base used for scale separation.
        n_scales (int): number of scales to compute the scattering over
        phi (Complex[Array, "h w"]): low pass output filter.
        psi1 (Complex[Array, "{n_scales} l h w"]): wavelet filters for first scattering field.
        psi2 (Complex[Array, "{n_scales} l h w"], optional): wavelet filters for 2nd scattering. Defaults to None (psi1 is re-used).
        strategy (str, optional): whether computation is done breadth- or depth-first down the scattering tree. Defaults to "breadth".

    Raises:
        ValueError: if specified computation strategy isn't "breadth" or "depth"
        ValueError: if image width or height isn't evenly divisible by adicity^(n_scales)

    Returns:
        List[Real[Array]]
    """
    if strategy not in ("depth", "breadth"):
        raise ValueError(
            'invalid computation strategy (`strategy`), must be "breadth" or "depth"'
        )
    if x.shape[-1] % adicity**n_scales:
        raise ValueError(
            "image width must be evenly divisible by adicity^(# scales)"
        )
    if x.shape[-2] % adicity**n_scales:
        raise ValueError(
            "image height must be evenly divisible by adicity^(# scales)"
        )
    max_scale = math.floor(math.log(min(x.shape[-2], x.shape[-1]), adicity))
    psi2 = psi1 if psi2 is None else psi2

    def scatter_coeff(z: Complex[Array, "..."]) -> Real[Array, "..."]:
        j = int(math.log(phi.shape[-1] / z.shape[-1], adicity))
        rest = max_scale - j
        return subsample_field(
            z * periodize_filter(phi, adicity, j)[None, None, ...],
            adicity,
            rest,
        ).squeeze(axis=(-2, -1))

    # first (zero-order) scattering coeff. is just low-pass'd input
    s0 = scatter_coeff(x)
    if strategy == "breadth":
        # \rho(x \conv psi_1)
        # compute wavelet convolution by pointwise mult. in fourier domain
        # multiplication should be elementwise(BC1HW x 11LHW => BCLHW)
        # output is sorted by j, each element corresp. with j-scale
        x1l = [
            complex_modulus(
                subsample_field(x, adicity, j)[:, :, None, ...]
                * periodize_filter(psi1[j, ...], adicity, j)[None, None, ...]
            )
            for j in range(n_scales)
        ]
        s1 = jnp.stack(list(map(scatter_coeff, x1l)), axis=2)

        # repeat process for 2nd layer by scattering the first scattering field
        s2 = []
        for j1, x1 in enumerate(x1l):
            # NOTE: we need to periodize the filter to a total scale of j1+j2
            # so that it matches in shape w. the field being (re-)subsampled
            # multiplication here is done between all pairs of angles l, m
            # x1 has shape : BCLHW -> BCL1HW
            # psi2 has shape : JMHW -> MHW -> 111MHW
            # output has shape : BCLMHW
            x2l = [
                scatter_coeff(
                    complex_modulus(
                        subsample_field(x1, adicity, j2)[:, :, :, None, ...]
                        * periodize_filter(psi2[j2, ...], adicity, j1 + j2)[
                            None, None, None, ...
                        ]
                    )
                )
                for j2 in range(j1 + 1, n_scales)
            ]
            if len(x2l) > 0:
                s2.append(x2l)
        s2 = list(map(Partial(jnp.stack, axis=2), s2))
    else:
        s1, s2 = [], []
        for j1 in range(n_scales):
            field = complex_modulus(
                subsample_field(x, adicity, j1)[:, :, None, ...]
                * periodize_filter(psi1[j1, ...], adicity, j1)[None, None, ...]
            )
            s1.append(scatter_coeff(field))
            s21 = []
            for j2 in range(j1 + 1, n_scales):
                s21.append(
                    scatter_coeff(
                        complex_modulus(
                            subsample_field(field, adicity, j2)[
                                :, :, :, None, ...
                            ]
                            * periodize_filter(psi2[j2, ...], adicity, j1 + j2)[
                                None, None, None, ...
                            ]
                        )
                    )
                )
            if len(s21):
                s2.append(s21)
        s1 = jnp.stack(s1, axis=2)
        s2 = list(map(Partial(jnp.stack, axis=2), s2))
    return s0, s1, s2


def complex_modulus(x: Complex[Array, "..."]) -> Real[Array, "..."]:
    """complex_modulus Compute modulus of scattering field.

    Args:
        x (Complex[Array]): input field.

    Returns:
        Real[Array]
    """
    return jnp.fft.rfft2(jnp.abs(jnp.fft.irfft2(x)))


def subsample_field(
    x: Complex[Array, "..."], adicity: int, j: int
) -> Complex[Array, "..."]:
    """subsample_field 2D subsampling of Fourier-domain image.

    Args:
        x (Complex[Array]): input array (4D)
        adicity (int): -adic downsampling factor (usually 2 for dyadic)
        j (int): factor to subsample each spatial dimension by

    Returns:
        Complex[Array]
    """
    if j == 0:
        return x
    k = adicity**j
    if len(x.shape) > 2:
        return jnp.mean(
            jnp.reshape(
                x,
                shape=list(x.shape[:-2])
                + [k, x.shape[-2] // k, k, x.shape[-1] // k],
            ),
            axis=(-4, -2),
        )
    else:
        return jnp.mean(
            jnp.reshape(x, shape=[k, x.shape[0] // k, k, x.shape[1] // k]),
            axis=(0, 2),
        )


def periodize_filter(
    x: Complex[Array, "..."], adicity: int, j: int
) -> Complex[Array, "..."]:
    """periodize_filter Crop filter(s) for multiplication with subsampled FTs.

    Args:
        x (Complex[Array]): input filter
        adicity (int): adicity of the spatial separation
        j (int): scale

    Returns:
        Complex[Array]: periodized filter
    """
    if len(x.shape) == 2:
        return _periodize(x, adicity, j)
    else:
        k = adicity**j
        inp_shape = x.shape
        rest = x.shape[:-2]
        out_shape = list(inp_shape)
        out_shape[-2] = out_shape[-2] // k
        out_shape[-1] = out_shape[-1] // k
        reshape = [math.prod(rest)] + [x.shape[-2], x.shape[-1]]
        return jnp.reshape(
            jnp.array(
                list(
                    map(
                        partial(_periodize, adicity=adicity, j=j),
                        jnp.reshape(x, shape=reshape),
                    )
                )
            ),
            shape=out_shape,
        )


def _periodize(
    x: Complex[Array, "h w"], adicity: int, j: int
) -> Complex[Array, "{h//adicity**j} {w//adicity**j}"]:
    k = adicity**j
    hgt, wid = x.shape[0], x.shape[1]

    crop = numpy.zeros((hgt // k, wid // k), x.dtype)

    mask = numpy.ones(x.shape, numpy.float32)
    len_x = math.floor(hgt * (1 - adicity ** (-j)))
    start_x = math.floor(hgt * adicity ** (-j - 1))
    len_y = math.floor(wid * (1 - adicity ** (-j)))
    start_y = math.floor(wid * adicity ** (-j - 1))
    mask[start_x : start_x + len_x, :] = 0
    mask[:, start_y : start_y + len_y] = 0
    x = numpy.multiply(x, mask)

    for _k in range(math.floor(hgt / k)):
        for _l in range(math.floor(wid / k)):
            for _i in range(k):
                for _j in range(k):
                    crop[_k, _l] += x[
                        _k + _i * math.floor(hgt / k),
                        _l + _j * math.floor(wid / k),
                    ]

    return crop
