""" Gabor wavelets.

Gabor wavelets are complex exponentials localized in space by a Gaussian.
"""

import math

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float


def _sigma_prefactor(bandwidth: float) -> float:
    """_sigma_prefactor prefactor for gabor filter.

    Args:
        bandwidth (float): bandwidth of filter

    Returns:
        float
    """
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return (
        1.0
        / jnp.pi
        * math.sqrt(math.log(2) / 2.0)
        * (2.0**bandwidth + 1)
        / (2.0**bandwidth - 1)
    )


def gabor_kernel_2d(
    frequency: float,
    theta: float = 0.0,
    bandwidth: float = 1.0,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    n_stds: int = 3,
    offset: float = 0,
    dtype=jnp.complex128,
) -> Complex[Array, " ..."]:
    """gabor_kernel_2d Complex 2D Gabor filter kernel.

    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Args:
        frequency (float): Spatial frequency of the harmonic function. Specified in pixels.
        theta (float): Orientation in radians. If 0, the harmonic is in the x-direction, optional. Defaults to 0.
        bandwidth (float): The bandwidth captured by the filter, optional. For fixed bandwidth, ``sigma_x`` and ``sigma_y`` will decrease with increasing frequency. This value is ignored if ``sigma_x`` and ``sigma_y`` are set by the user.
        sigma_x (float): Standard deviation in x-direction, before rotation.
        sigma_y (float): Standard deviation in y-direction, before rotation.
        n_stds (int): The linear size of the kernel is `n_stds` standard deviations, optional. Defaults to 3.
        offset (float): Phase offset of harmonic function in radians, optional. Defaults to 0.
        dtype : Specifies if the filter is single or double precision complex. One of or {np.complex64, np.complex128}.

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
        + 1j * (2 * jnp.pi * frequency * rotx + offset)
    ) * (1 / (2 * jnp.pi * sigma_x * sigma_y))


def gabor_kernel_3d(
    frequency: float,
    theta: float = 0.0,
    psi: float = 0.0,
    bandwidth: float = 1.0,
    sigma_x: float | None = None,
    sigma_y: float | None = None,
    sigma_z: float | None = None,
    n_stds: int = 3,
    offset=0,
    dtype=jnp.complex128,
) -> Complex[Array, " ..."]:
    """gabor_kernel_3d Complex 3D Gabor filter kernel.

    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Args:
        frequency (float): Spatial frequency of the harmonic function. Specified in pixels.
        theta (float): Orientation in radians. If 0, the harmonic is in the x-direction, optional. Defaults to 0.
        bandwidth (float): The bandwidth captured by the filter, optional. For fixed bandwidth, ``sigma_x`` and ``sigma_y`` will decrease with increasing frequency. This value is ignored if ``sigma_x`` and ``sigma_y`` are set by the user.
        sigma_x (float): Standard deviation in x-direction, before rotation.
        sigma_y (float): Standard deviation in y-direction, before rotation.
        sigma_z (float): Standard deviation in z-direction, before rotation.
        n_stds (int): The linear size of the kernel is `n_stds` standard deviations, optional. Defaults to 3.
        offset (float): Phase offset of harmonic function in radians, optional. Defaults to 0.
        dtype : Specifies if the filter is single or double precision complex. One of or {np.complex64, np.complex128}.

    Returns:
        Complex[Array]
    """
    raise NotImplementedError("todo")


def _rotation_matrix(theta: float, psi: float) -> Float[Array, "3 3"]:
    """rotation_matrix 3D rotation matrix.

    Args:
        theta (float) : rotation angle about x-axis
        psi (float): rotation angle about z-axis

    Returns:
        Float[Array]: 3x3 rotation matrix
    """
    rot_x = jnp.asarray(
        [
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ]
    )
    rot_z = jnp.asarray(
        [
            [math.cos(psi), -math.sin(psi), 0],
            [math.sin(psi), math.cos(psi), 0],
            [0, 0, 1],
        ]
    )
    return rot_x @ rot_z
