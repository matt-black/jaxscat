""" Fourier-space 2D Scattering Transform.

Scattering transform for 2 dimensions, where all calculations are done in
the Fourier domain.
"""

import math
from functools import partial
from typing import Callable
from typing import List
from typing import Tuple

import jax.numpy as jnp
import numpy
from jax.tree_util import Partial
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Real


def complex_modulus(x: Complex[Array, "..."]) -> Complex[Array, "..."]:
    """complex_modulus Compute modulus of scattering field.

    Args:
        x (Complex[Array]): input field.

    Returns:
        Real[Array]
    """
    return jnp.fft.fft2(jnp.abs(jnp.fft.ifft2(x)))


def scattering_fields(
    x: Complex[Array, "b c h w"],
    adicity: int,
    n_scales: int,
    psi1: Complex[Array, "{n_scales} l h w"],
    psi2: Complex[Array, "{n_scales} l h w"] | None = None,
    nonlinearity: Callable[[Array], Array] = complex_modulus,
) -> Tuple[List[Complex[Array, "..."]], List[List[Complex[Array, "..."]]]]:
    """scattering_fields Compute scattering fields.

    Args:
        x (Complex[Array]): input array, should be 4D (BCHW)
        adicity (int): adicity of scale separation
        n_scales (int): number of scales to compute fields over
        psi1 (Complex[Array]): wavelet filter bank for first layer
        psi2 (Complex[Array], optional): wavelet filter bank for second layer
        nonlinearity (Callable[[Array],Array], optional): nonlinearity to use after each wavelet transform. Defaults to complex modulus.
    Raises:
        ValueError: if image width or height isn't evenly divisible by adicity^(n_scales)

    Returns:
        Tuple[List[Complex[Array]],List[List[Complex[Array]]]]: tuple where first element is list of fields at first layer of transform, second is list of lists of fields at second layer of transform.
    """
    if x.shape[-1] % adicity**n_scales:
        raise ValueError(
            "image width must be evenly divisible by adicity^(# scales)"
        )
    if x.shape[-2] % adicity**n_scales:
        raise ValueError(
            "image height must be evenly divisible by adicity^(# scales)"
        )
    psi2 = psi1 if psi2 is None else psi2
    # each field has shape BCLHW
    # will need to concat BCLMHW, so transform BCL1HW
    field1 = [
        nonlinearity(
            subsample_field(x, adicity, j)[:, :, None, ...]
            * periodize_filter(psi1[j, ...], adicity, j)[None, None, ...]
        )
        for j in range(n_scales)
    ]
    field2 = []
    for j1, x1 in enumerate(field1[:-1]):
        # NOTE: we need to periodize the filter to a total scale of j1+j2
        # so that it matches in shape w. the field being (re-)subsampled
        # multiplication here is done between all pairs of angles l, m
        # psi2 has shape : JMHW -> MHW -> 111MHW
        # output has shape : BCLMHW
        j2s = list(range(j1 + 1, n_scales))
        x2l = [
            nonlinearity(
                subsample_field(x1, adicity, j2)[:, :, :, None, ...]
                * periodize_filter(psi2[j2, ...], adicity, j1 + j2)[
                    None, None, None, ...
                ]
            )
            for j2 in j2s
        ]
        field2.append(x2l)
    field1 = [f[:, :, :, None, ...] for f in field1]
    return field1, field2


def scattering_coeffs(
    x: Real[Array, "b c h w"],
    adicity: int,
    n_scales: int,
    phi: Complex[Array, "h w"],
    psi1: Complex[Array, "{n_scales} l h w"],
    psi2: Complex[Array, "{n_scales} l h w"] | None = None,
    strategy: str = "breadth",
    reduction: str = "local",
    nonlinearity: Callable[[Array], Array] = complex_modulus,
) -> Tuple[Real[Array, "..."], Real[Array, "..."], Real[Array, "..."]]:
    """scattering_coeffs Compute scattering coefficients for input field, `x`.

    Args:
        x (Complex[Array, "batch channel height width"]): 4D input tensor. Coefficients are computed on last 2 dimensions. Should be Fourier domain.
        adicity (int): logarithm-base used for scale separation.
        n_scales (int): number of scales to compute the scattering over
        phi (Complex[Array, "h w"]): low pass output filter.
        psi1 (Complex[Array, "{n_scales} l h w"]): wavelet filters for first scattering field.
        psi2 (Complex[Array, "{n_scales} l h w"], optional): wavelet filters for 2nd scattering. Defaults to None (psi1 is re-used).
        strategy (str, optional): whether computation is done breadth- or depth-first down the scattering tree. Defaults to "breadth".
        reduction (str, optional): how to reduce the fields to scattering coefficients. Either "local" or "global". Defaults to "local".
        nonlinearity (Callable[[Array],Array], optional): nonlinearity to use after each wavelet transform. Defaults to complex modulus.
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
    if reduction not in ("global", "local"):
        raise ValueError(
            'invalid reduction strategy (`reduction`) must be "global" or "local"'
        )
    if x.shape[-1] % adicity**n_scales:
        raise ValueError(
            "image width must be evenly divisible by adicity^(# scales)"
        )
    if x.shape[-2] % adicity**n_scales:
        raise ValueError(
            "image height must be evenly divisible by adicity^(# scales)"
        )
    psi2 = psi1 if psi2 is None else psi2

    def scatter_coeff(z: Complex[Array, "..."]) -> Real[Array, "..."]:
        j = int(math.log(phi.shape[-1] / z.shape[-1], adicity))
        rest = n_scales - j
        if rest > 0:
            out = jnp.fft.ifft2(
                subsample_field(
                    z * periodize_filter(phi, adicity, j)[None, None, ...],
                    adicity,
                    rest,
                )
            ).real
        else:
            out = jnp.fft.ifft2(
                z * periodize_filter(phi, adicity, j)[None, None, ...]
            ).real
        # if the inputs were padded, they won't be evenly divisible by (2**n_scales-1)
        # if padded, need to crop out 2 pixels/dim, one on each "side" of dimension
        if out.shape[-2] == x.shape[-2] // adicity ** (n_scales - 1):
            slc_h = slice(None)
        else:
            slc_h = slice(1, -1)
        if out.shape[-1] == x.shape[-1] // adicity ** (n_scales - 1):
            slc_w = slice(None)
        else:
            slc_w = slice(1, -1)
        # local reduction just returns the local scattering coeffs,
        if reduction == "local":
            return out[..., slc_h, slc_w]
        else:  # global reduction takes mean over space dimensions
            return jnp.mean(out[..., slc_h, slc_w], axis=(-2, -1))

    # first (zero-order) scattering coeff. is just low-pass'd input
    s0 = scatter_coeff(x)
    if strategy == "breadth":
        # \rho(x \conv psi_1)
        # compute wavelet convolution by pointwise mult. in fourier domain
        # multiplication should be elementwise(BC1HW x 11LHW => BCLHW)
        # output is sorted by j, each element corresp. with j-scale
        x1l = [
            nonlinearity(
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
                    nonlinearity(
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
            field = nonlinearity(
                subsample_field(x, adicity, j1)[:, :, None, ...]
                * periodize_filter(psi1[j1, ...], adicity, j1)[None, None, ...]
            )
            s1.append(scatter_coeff(field))
            s21 = []
            for j2 in range(j1 + 1, n_scales):
                s21.append(
                    scatter_coeff(
                        nonlinearity(
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
