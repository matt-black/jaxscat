""" Morlet wavelets.

Morlet filters are Gabor filters modified such that they can function as bandpass filters.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Real

from .gabor import gabor_kernel_2d_real


def filter_bank_2d(
    size_h: int,
    size_w: int,
    n_orientations: int,
    n_scales: int,
    space: str,
    adicity: int = 2,
    sigma_prefactor: float = 0.8,
    freq_prefactor: float = 3 * math.pi / 4,
    gamma_prefactor: float = 4.0,
):
    """filter_bank_2d Filter bank of 2D Morlet filters.

    Args:
        size_h (int): spatial size of filter, height. Specified in pixels.
        size_w (int): Spatial size of filter, width. Specified in pixels.
        n_orientations (int): Number of orientations in the filter bank.
        n_scales (int): Number of spatial scales in the filter bank.
        space (str): Output space of filters, one of 'real' or 'fourier'.
        adicity (int, optional): Adicity of the spatial scale separation. Defaults to 2.
        sigma_prefactor (float, optional): sigma = (sigma_prefactor) * adicity**scale. Defaults to 0.8.
        freq_prefactor (float, optional): freq = (freq_prefactor) / adicity**scale. Defaults to 3*math.pi/4.
        gamma_prefactor (float, optional): gamma = gamma_prefactor / n_orientations. Defaults to 4.0.

    Raises:
        ValueError: if output space isn't one of 'real' or 'fourier'

    Returns:
        ndarray: real if output space is 'fourier', complex if output space is 'real'.
    """
    sigmas = [sigma_prefactor * adicity**j for j in range(n_scales)]
    freqs = [freq_prefactor / adicity**j for j in range(n_scales)]
    thetas = [
        math.floor(n_orientations - n_orientations / 2 - ell)
        * math.pi
        / n_orientations
        for ell in range(n_orientations)
    ]
    if space == "fourier":
        kernel_fun = morlet_kernel_2d_fourier
    elif space == "real":
        kernel_fun = morlet_kernel_2d_real
    else:
        raise ValueError('invalid output space, must be "real" or "fourier"')
    return jnp.stack(
        [
            jnp.stack(
                [
                    kernel_fun(
                        size_h,
                        size_w,
                        sigma,
                        freq,
                        theta,
                        gamma_prefactor / n_orientations,
                    )
                    for theta in thetas
                ],
                axis=0,
            )
            for sigma, freq in zip(sigmas, freqs)
        ],
        axis=0,
    )


def morlet_kernel_2d_real(
    size_h: int,
    size_w: int,
    sigma: float,
    frequency: float,
    theta: float,
    gamma: float = 1.0,
    offset: float = 0.0,
    dtype=jnp.complex64,
) -> Complex[Array, "{size_h} {size_w}"]:
    """morlet_kernel_2d Real-space kernel for 2D Morlet filter.

    Args:
        size_h (int): spatial size of filter, height, in pixels.
        size_w (int): spatial size of filter, width, in pixels.
        sigma (float) : bandwidth of filter (linear size of receptive field).
        frequency (float): spatial frequency of the cosine factor
        theta (float): angle of filter, in [0, pi]
        gamma (float, optional): ellipticity of the filter. Defaults to 1.0.
        offset (float, optional): phase offset of the cosine factor. Defaults to 0.0.
        dtype : datatype of output. One of `jnp.complex64` or `jnp.complex128`. Defaults to jnp.complex64.

    Returns:
        Complex[Array]
    """
    gab = gabor_kernel_2d_real(
        size_h, size_w, sigma, frequency, theta, gamma, offset, dtype
    )
    mod = gabor_kernel_2d_real(
        size_h, size_w, sigma, 0.0, theta, gamma, offset, dtype
    )
    ratio = jnp.sum(gab) / jnp.sum(mod)
    return gab - ratio * mod


def morlet_kernel_2d_fourier(
    size_h: int,
    size_w: int,
    sigma: float,
    frequency: float,
    theta: float,
    gamma: float = 1.0,
    offset: float = 0.0,
    dtype=jnp.float32,
) -> Real[Array, "{size_h} {size_w}"]:
    """morlet_kernel_2d Real-space kernel for 2D Morlet filter.

    Args:
        size_h (int): spatial size of filter, height, in pixels.
        size_w (int): spatial size of filter, width, in pixels.
        sigma (float) : bandwidth of filter (linear size of receptive field).
        frequency (float): spatial frequency of the cosine factor
        theta (float): angle of filter, in [0, pi]
        gamma (float, optional): ellipticity of the filter. Defaults to 1.0.
        offset (float, optional): phase offset of the cosine factor. Defaults to 0.0.
        dtype : datatype of output. One of `jnp.complex64` or `jnp.complex128`. Defaults to jnp.complex64.

    Returns:
        Real[Array]
    """
    cdtype = jnp.complex64 if dtype == jnp.float32 else jnp.complex128
    return jnp.real(
        jnp.fft.fft2(
            morlet_kernel_2d_real(
                size_h, size_w, sigma, frequency, theta, gamma, offset, cdtype
            )
        )
    )
