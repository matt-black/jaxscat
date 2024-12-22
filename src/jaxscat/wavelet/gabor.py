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


def _sigma_prefactor(bandwidth: float) -> float:
    b = bandwidth
    return math.sqrt(math.log(2) / 2.0) / jnp.pi * (2.0**b + 1) / (2.0**b - 1)


def gabor_kernel_2d_real_scikit(
    frequency: float,
    theta: float = 0.0,
    bandwidth: float = 1.0,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    n_stds: float = 3,
    offset: float = 0,
    square: bool = False,
    dtype=jnp.complex64,
) -> Complex[Array, "h w"]:
    """gabor_kernel_2d_real_scikit Real-space 2D Gabor kernel.

    Args:
        frequency (float): Spatial frequency of the harmonic function, in pixels.
        theta (float, optional): Orientation in radians. Defaults to 0..
        bandwidth (float, optional): Filter bandwidth, sets sigma_x,y. Defaults to 1..
        sigma_x (float | None, optional): Standard deviation in x-direction (pre-rotation). Defaults to None.
        sigma_y (float | None, optional): Standard deviation in y-direction (pre-rotation). Defaults to None.
        n_stds (float, optional): Size of the output kernel, in standard deviations. Defaults to 3.
        offset (float, optional): Phase offset of the harmonic function. Defaults to 0.
        square (bool, optional): Make the output kernel square. Defaults to False.
        dtype (optional): Datatype of output kernel (single or double precision). Defaults to jnp.complex128.

    Raises:
        ValueError: If output dtype is not complex

    Returns:
        Complex[Array]
    """
    if sigma_x is None:
        sigma_x = _sigma_prefactor(bandwidth) / frequency
    if sigma_y is None:
        sigma_y = _sigma_prefactor(bandwidth) / frequency

    if jnp.dtype(dtype).kind != "c":
        raise ValueError("dtype must be complex")

    ct, st = math.cos(theta), math.sin(theta)
    x0 = math.ceil(
        max(abs(n_stds * sigma_x * ct), abs(n_stds * sigma_y * st), 1)
    )
    y0 = math.ceil(
        max(abs(n_stds * sigma_y * ct), abs(n_stds * sigma_x * st), 1)
    )
    if square:
        if x0 > y0:
            y0 = x0
        if y0 > x0:
            x0 = y0
    y, x = jnp.meshgrid(
        jnp.arange(-y0, y0 + 1),
        jnp.arange(-x0, x0 + 1),
        indexing="ij",
        sparse=True,
    )
    rotx = x * ct + y * st
    roty = -x * st + y * ct

    return jnp.exp(
        -0.5 * (rotx**2 / sigma_x**2 + roty**2 / sigma_y**2)
        + 1j * (2 * jnp.pi * frequency * rotx + offset),
    ) / (2 * jnp.pi * sigma_x * sigma_y)
