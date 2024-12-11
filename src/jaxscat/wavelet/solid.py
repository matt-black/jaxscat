r""" Solid Harmonic Wavelets

First introduced in Eickenberg, et al [1]:
"Solid harmonics are solutions of the Laplace equation, $\Delta{}f=0$."
Wavelets are obtained by localizing the support of a solid harmonic in space
by multiplication with a Gaussian.

In 2D: $\psi_\ell(r,\phi) =
"""

import jax.lax
import jax.numpy as jnp
from jaxtyping import Float


def phi_ell(ell: float, r: float, phi: float) -> float:
    return (
        (1 / jnp.sqrt(4 * jnp.pi**2))
        * jnp.exp(-0.5 * r**2)
        * r**ell
        * jnp.exp(jax.lax.complex(0, ell * phi))
    )
