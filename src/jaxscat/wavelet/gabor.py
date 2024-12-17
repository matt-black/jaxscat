""" Gabor wavelets.

Gabor wavelets are fourier modes modulated by a gaussian envelope.
The gaussian envelope localizes the fourier mode in space.
"""

import math
from itertools import product

import jax
import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Real


def gabor_kernel_2d_real(
    size_h: int,
    size_w: int,
    sigma: float,
    frequency: float,
    theta: float,
    gamma: float = 1.0,
    offset: float = 0.0,
    dtype=jnp.complex64,
) -> Complex[Array, "{size_h} {size_w}"]:
    """gabor_kernel_2d Real-space kernel for 2D Gabor filter.

    Args:
        size_h (int): spatial size of filter, height, in pixels.
        size_w (int): spatial size of filter, width, in pixels.
        sigma (float) : bandwidth of filter (linear size of receptive field).
        frequency (float): spatial frequency of the cosine factor
        theta (float): angle of filter, in [0, pi]
        gamma (float, optional): ellipticity of the filter. Defaults to 1.0.
        offset (float, optional): phase offset of the cosine factor. Defaults to 0.0.
        dtype : datatype of output. One of `jnp.complex64` or `jnp.complex128`. Defaults to jnp.complex64.
    """
    rot = jnp.asarray(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ]
    )
    scl = jnp.asarray([[1, 0], [0, gamma**2]])
    rot_inv = jnp.asarray(
        [
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)],
        ]
    )
    t = (rot @ scl @ rot_inv) / (2 * sigma**2)
    v = jnp.asarray([math.cos(theta), math.sin(theta)]) * frequency

    out = jnp.zeros((size_h, size_w), dtype=dtype)
    for _x, _y in product(range(-2, 3), range(-2, 3)):
        sx, sy = _x * size_w + offset, _y * size_h + offset
        xy = jnp.stack(
            jnp.meshgrid(
                jnp.arange(sx, sx + size_w),
                jnp.arange(sy, sy + size_h),
                indexing="ij",
            ),
            axis=0,
        )
        out += jnp.exp(
            jax.lax.complex(
                -jnp.einsum(
                    "ij,ij...->...", t, jnp.einsum("i...,j...->ij...", xy, xy)
                ),
                jnp.einsum("i,i...->...", v, xy),
            )
        )
    return out / (2 * math.pi * sigma**2 / gamma)


def gabor_kernel_2d_fourier(
    size_h: int,
    size_w: int,
    sigma: float,
    frequency: float,
    theta: float,
    gamma: float = 1.0,
    offset: float = 0.0,
    dtype=jnp.float32,
) -> Real[Array, "{size_h} {size_w}"]:
    """gabor_kernel_2d Fourier-space kernel for 2D Gabor filter.

    Args:
        size_h (int): spatial size of filter, height, in pixels.
        size_w (int): spatial size of filter, width, in pixels.
        sigma (float) : bandwidth of filter (linear size of receptive field).
        frequency (float): spatial frequency of the cosine factor
        theta (float): angle of filter, in [0, pi]
        gamma (float, optional): ellipticity of the filter. Defaults to 1.0.
        offset (float, optional): phase offset of the cosine factor. Defaults to 0.0.
        dtype : datatype of output. One of `jnp.complex64` or `jnp.complex128`. Defaults to jnp.complex64.
    """
    cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    return jnp.real(
        jnp.fft.fft2(
            gabor_kernel_2d_real(
                size_h, size_w, sigma, frequency, theta, gamma, offset, cdtype
            )
        )
    )
