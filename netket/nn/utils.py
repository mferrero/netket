from functools import partial, lru_cache
from typing import Union, Tuple, Optional
import math

import numpy as np
from jax import numpy as jnp
from netket.utils import get_afun_if_module, mpi
from netket.utils.types import Array
from netket.hilbert import DiscreteHilbert
import jax
from flax.traverse_util import flatten_dict, unflatten_dict
from flax.core import unfreeze


def split_array_mpi(array):
    """
    Splits the first dimension of the input array among mpi processes.
    Works like `mpi.scatter`, but assumes that the input array is available and
    identical on all ranks.
    !!! Warn
         The output is a numpy array.
    Args:
         array: A nd-array

    Result:
        A numpy array, of potentially different state on every mpi rank.
    """

    n_states = array.shape[0]
    states_n = np.arange(n_states)

    # divide the hilbert space in chunks for each node
    states_per_rank = np.array_split(states_n, mpi.n_nodes)

    return array[states_per_rank[mpi.rank]]


def to_array(hilbert, apply_fun, variables, normalize=True, allgather=True):
    """
    Computes `apply_fun(variables, states)` on all states of `hilbert` and returns
      the results as a vector.

    Args:
        normalize: If True, the vector is normalized to have L2-norm 1.
        allgather: If True, the final wave function is stored in full at all MPI ranks.
    """
    if not hilbert.is_indexable:
        raise RuntimeError("The hilbert space is not indexable")

    apply_fun = get_afun_if_module(apply_fun)

    # mpi4jax does not have (yet) allgatherv so we need to be creative
    # could be made easier if we update mpi4jax
    n_states = hilbert.n_states
    n_states_padded = int(np.ceil(n_states / mpi.n_nodes)) * mpi.n_nodes
    states_n = np.arange(n_states)
    fake_states_n = np.arange(n_states_padded - n_states)

    # divide the hilbert space in chunks for each node
    states_per_rank = np.split(np.concatenate([states_n, fake_states_n]), mpi.n_nodes)

    xs = hilbert.numbers_to_states(states_per_rank[mpi.rank])

    return _to_array_rank(apply_fun, variables, xs, n_states, normalize, allgather)


@partial(jax.jit, static_argnums=(0, 3, 4, 5))
def _to_array_rank(apply_fun, variables, σ_rank, n_states, normalize, allgather):
    """
    Computes apply_fun(variables, σ_rank) and gathers all results across all ranks.
    The input σ_rank should be a slice of all states in the hilbert space of equal
    length across all ranks because mpi4jax does not support allgatherv (yet).

    Args:
        n_states: total number of elements in the hilbert space.
    """
    # number of 'fake' states, in the last rank.
    n_fake_states = σ_rank.shape[0] * mpi.n_nodes - n_states

    log_psi_local = apply_fun(variables, σ_rank)

    # last rank, get rid of fake elements
    if mpi.rank == mpi.n_nodes - 1 and n_fake_states > 0:
        log_psi_local = log_psi_local.at[-n_fake_states:].set(-jnp.inf)

    if normalize:
        # subtract logmax for better numerical stability
        logmax, _ = mpi.mpi_max_jax(log_psi_local.real.max())
        log_psi_local -= logmax

    psi_local = jnp.exp(log_psi_local)

    if normalize:
        # compute normalization
        norm2 = jnp.linalg.norm(psi_local) ** 2
        norm2, _ = mpi.mpi_sum_jax(norm2)
        psi_local /= jnp.sqrt(norm2)

    if allgather:
        psi, _ = mpi.mpi_allgather_jax(psi_local)
    else:
        psi = psi_local

    psi = psi.reshape(-1)

    # remove fake states
    psi = psi[0:n_states]
    return psi


def to_matrix(hilbert, machine, params, normalize=True):

    if not hilbert.is_indexable:
        raise RuntimeError("The hilbert space is not indexable")

    psi = to_array(hilbert, machine, params, normalize=False)

    L = hilbert.physical.n_states
    rho = psi.reshape((L, L))
    if normalize:
        trace = jnp.trace(rho)
        rho /= trace

    return rho


# TODO: Deprecate: remove
def update_dense_symm(params, names=["dense_symm", "Dense"]):
    """Updates DenseSymm kernels in pre-PR#1030 parameter pytrees to the new
    3D convention.

    Args:
        params: a parameter pytree
        names: layer names search for, default: those used in RBMSymm and GCNN*
    """
    params = unfreeze(params)  # just in case, doesn't break with a plain dict

    def fix_one_kernel(args):
        path, array = args
        if (
            len(path) > 1
            and path[-2] in names
            and path[-1] == "kernel"
            and array.ndim == 2
        ):
            array = jnp.expand_dims(array, 1)
        return (path, array)

    return unflatten_dict(dict(map(fix_one_kernel, flatten_dict(params).items())))


@lru_cache(maxsize=64)
def _get_output_idx(
    shape: Tuple[int, ...], max_bits: Optional[int] = None, doubled: bool = False
) -> Tuple[Tuple[int, ...], int]:
    bits_per_local_occupation = (
        tuple(np.ceil(np.log2(shape)).astype(int)) * 2 if doubled else 1
    )
    if max_bits is None:
        max_bits = max(bits_per_local_occupation)
    output_idx = []
    offset = 0
    for b in bits_per_local_occupation:
        output_idx.extend([i + offset for i in range(b)])
        offset += max_bits
    output_idx = tuple(output_idx)
    return output_idx, max_bits


def binary_encoding(
    shape: Union[DiscreteHilbert, Tuple[int, ...]],
    x: Array,
    *,
    max_bits: Optional[int] = None,
    doubled: bool = False,
) -> Array:
    """
    Encodes the array `x` into a set of binary-encoded variables described by shape.
    """
    if isinstance(shape, DiscreteHilbert):
        shape = shape.shape
    jax.core.concrete_or_error(None, shape, "Shape must be known statically")
    max_bits, output_idx = _get_output_idx(shape, max_bits, doubled)
    return _binarise(x, max_bits, output_idx)


def _binarise_loop(i, s):
    substates = s["states"][..., i].astype(int)[..., jnp.newaxis]
    s["binarised_states"] = (
        s["binarised_states"]
        .at[..., i, :]
        .set(
            substates & 2 ** jnp.arange(s["binarised_states"].shape[-1], dtype=int) != 0
        )
        .astype(s["states"].dtype)
    )
    return s


@partial(jax.jit, static_argnums=[1, 2])
def _binarise(states, max_bits, output_idx):
    init_s = {
        "binarised_states": jnp.empty(states.shape + (max_bits,), dtype=states.dtype),
        "states": states,
    }
    s = jax.lax.fori_loop(0, states.shape[-1], _binarise_loop, init_s)
    binarised_states = s["binarised_states"]
    binarised_states = binarised_states.reshape(
        *binarised_states.shape[:-2], math.prod(binarised_states.shape[-2:])
    )
    return binarised_states[..., output_idx]
