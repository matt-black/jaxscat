""" Private utilities. For internal library use only.

To expose a utility function via the public API, import it in `util.py`
"""

import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array
from jaxtyping import Complex
from jaxtyping import Float
from jaxtyping import Real


def binomial_coefficient(
    x: Real[Array, "..."] | Real, y: Real[Array, "..."] | Real
) -> Float[Array, "..."]:
    """binomial_coefficient binomial coefficient as a function of 2 variables.

    Returns:
        Real[Array,"..."]|Real: value of binomial coefficient
    """
    return jsp.special.gamma(x + 1) / (
        jsp.special.gamma(y + 1) * jsp.special.gamma(x - y + 1)
    )


def ifft2_centered(centered_fft: Complex[Array, "a a"]) -> Float[Array, "a a"]:
    """ifft2_centered inverse fft of a centered fft.

    Returns:
        Float[Array]
    """
    cent = centered_fft.shape[0] // 2
    # move 0 freq to top left, then take the ifft and retranslate to middle
    fft = jnp.roll(centered_fft, (-cent, -cent), (0, 1))
    im = jnp.fft.ifft2(fft)
    return jnp.roll(im, (cent, cent), (0, 1))


# polar coordinate grid generation
def radial_coordinate_grid_2d(size: int) -> Float[Array, "{size} {size}"]:
    """radial_coordinate_grid_2d 2d grid of radial coordinates for x,y.

    r = sqrt(x^2 + y^2)

    Args:
        size (int): size of single dimension of grid

    Returns:
        Float[Array]: 2d grid of radial coordinate values
    """
    cent = size / 2
    x, y = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    return jnp.sqrt(jnp.power(x - cent, 2) + jnp.power(y - cent, 2))


def angular_coordinate_grid_2d(size: int) -> Float[Array, "{size} {size}"]:
    """angular_coordinate_grid_2d 2d grid of angles for each x,y in grid.

    a = atan2(y/x) where y and x are based on the grid center being (0,0)

    Args:
        size (int): size of single dimension of grid

    Returns:
        Float[Array]: 2d square grid of angular values at each coordinate
    """
    cent = size / 2
    x, y = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    return jnp.arctan2(y - cent, x - cent)


def shift_remainder(v: Float[Array, "..."]) -> Float[Array, "..."]:
    """shift_remainder TODO.

    Args:
        v (Float[Array]): input values (angles)

    Returns:
        Float[Array]
    """
    return jnp.remainder(v + jnp.pi, 2 * jnp.pi) - jnp.pi
