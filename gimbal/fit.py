from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jit
from jax.scipy.special import logsumexp
from tensorflow_probability.substrates.jax.distributions import VonMisesFisher as VMF

@jit
def hornik_mle(rbars, dim):
    nu = dim/2 - 1

    def G(kappa, a, b):
        return kappa / (a + jnp.sqrt(kappa**2 + b**2))

    def Ginv(rho, a, b):
        """Equation (4)"""
        return rho/(1-rho**2) \
                * (a + jnp.sqrt((rho**2)*(a**2) + (1-rho**2)*(b**2)))

    Ginv_ub = partial(Ginv, a=nu+0.5, b=nu+1.5)
    Ginv_lb_0 = partial(Ginv, a=nu, b=nu+2)
    Ginv_lb_1 = partial(Ginv, a=nu+0.5, b=jnp.sqrt((nu+0.5)*(nu+1.5)))

    ub = Ginv_ub(rbars)
    lb = jnp.maximum(Ginv_lb_0(rbars), Ginv_lb_1(rbars))
    return (ub+lb)/2


def _calculate_J_dir(sigmasq, parents, dim):
    """Generates precision matrix prior of a multivariate-Gaussian whose
    components are nodes of a tree-structured graph,
    i.e. a graph Laplacian matrix. Assumes isotropic noise in d=2 space

    Parameters
        sigmasq: array_like, length n
        parents: array_like, length n
        dim: integer
    Returns
        G: ndarray, (n*dim, n*dim)
    """
    n = len(parents)

    G = jnp.zeros((n*dim, n*dim))
    for i in range(n):
        # On-diagonal: self-iteraction
        G = G.at[i*dim:(i+1)*dim, i*dim:(i+1)*dim].add(
            1/sigmasq[i] * jnp.eye(dim))
                      
        for k in range(i+1, n):
            if parents[k] == i:  # For children of i:
                # On-diagonal: Add an extra degree term
                G = G.at[i*dim:(i+1)*dim, i*dim:(i+1)*dim].add(
                    1/sigmasq[k] * jnp.eye(dim))
                
                # Off-diagonal: -1 * adjacency term
                G = G.at[i*dim:(i+1)*dim, k*dim:(k+1)*dim].add(
                    -1/sigmasq[k] * jnp.eye(dim))

                G = G.at[k*dim:(k+1)*dim, i*dim:(i+1)*dim].add(
                    -1/sigmasq[k] * jnp.eye(dim))  
    return G

def _rotate_u_dir(dir, headings):
    """Rotates direction vectors in canonical ref frame to absolute ref frame
        dir: ndarray, (..., num_keypts, dim)
        headings: ndarray, (..., )
    """

    Rs = _rotation_mat(headings)
    
    return jnp.einsum('tyx, tjx-> tjy', Rs, dir)

def _calculate_h_dir(dir, children, radii, sigmasq):
    """
    Calculates information bias for given by direction vectors
    Parameters
        directions: ndarray, shape (..., num_keypts, dim)
        parents, radii, sigmasq: array_like, length num_keypts
    """

    num_keypts, dim = dir.shape[-2:]

    h = dir * radii[:,None] / sigmasq[:,None]
    for j in range(num_keypts):    
        for k in children[j]:  # Each child subtracts a term
            h = h.at[..., j,:].add(-dir[...,k,:] * radii[k] / sigmasq[k])
    return h

@jit
def e_step(xs, pis, mus, kappas):
    """
    Parameters
    ----------
        xs:     shape (num_timesteps, num_joints, dim)
            Observed direction vectors, canonical space
        pis:    shape (num_states, )
        mus:    shape (num_states, num_joints, dim)
        kappas: shape (num_states, num_joints)

    Returns
    -------
        E_zs: shape (num_timesteps, num_states)
    """
    MIN_PROB = 1e-6
    
    # VMF(mus, kappas): batch_shape (num_states, num_joints), event_shape (dim,)
    # lps: shape(num_timesteps, num_states)
    lps = jnp.sum(VMF(mus, kappas).log_prob(xs[:,None,:,:]), axis=-1) + jnp.log(pis)

    # shape (num_timesteps)
    log_norm = logsumexp(lps, axis=-1, keepdims=True)
    E_zs = jnp.exp(lps - log_norm) + MIN_PROB
    return E_zs / E_zs.sum(1, keepdims=True)



@jit
def ll_movMF(xs, pis, mus, kappas):
    """
    Parameters
    ----------
        xs:     shape (num_timesteps, num_joints, dim)
            Observed direction vectors
        pis:    shape (num_states, )
        mus:    shape (num_states, num_joints, dim)
        kappas: shape (num_states, num_joints)

    Returns
    -------
        lls:    shape (num_timesteps)
    """
    lps = jnp.sum(VMF(mus, kappas).log_prob(xs[:,None]), axis=-1) + jnp.log(pis)
    return logsumexp(lps, axis=-1)

@jit
def m_step(xs, E_zs):
    """
    Parameters
    ----------
        xs:   shape (num_timesteps, num_joints, dim)
            Observed direction vectors
        E_zs: shape (num_timesteps, num_states)

    Returns
    -------
        pis_mle:    shape (num_states, )
        mus_mle:    shape (num_states, num_joints, dim)
        kappas_mle: shape (num_states, num_joints)
    """
    MAX_KAPPA = 100
    
    # MLE of pi parameter
    # -------------------
    pis_mle = jnp.sum(E_zs, axis=0) / len(E_zs)
    
    # MLE of mu parameter
    # -------------------
    # shape (num_states, num_joints, dim)
    Rs = jnp.sum(E_zs[:,:,None,None] * xs[:, None, :,:], axis=0)
    mus_mle = Rs / jnp.linalg.norm(Rs, axis=-1, keepdims=True)
    
    # MLE of kappa parameter
    # -------------------
    # Mean resultant length, shape (num_states, num_joints)
    rbars = jnp.minimum(jnp.linalg.norm(Rs, axis=-1) / jnp.sum(E_zs, axis=0)[:,None], 1-1e-10)
    kappas_mle = jnp.minimum(hornik_mle(rbars, xs.shape[-1]), MAX_KAPPA)
    
    return pis_mle, mus_mle, kappas_mle

@jit
def em_step(t, carry):
    _, pis, mus, kappas, lls, xs = carry

    E_zs = e_step(xs, pis, mus, kappas)
    pis, mus, kappas = m_step(xs, E_zs)
    
    # Compute log likelihood
    lls = lls.at[t].set(jnp.sum(ll_movMF(xs, pis, mus, kappas)))

    return (E_zs, pis, mus, kappas, lls, xs)
    
def em_movMF(key, xs, num_states, num_iters):
    """
    xs should not pass in joint j=0, so we don't have to worry about it being undefined
    xs: shape (num_timesteps, num_joints, dim)
    """
    num_timesteps, num_joints, dim = xs.shape

    # Initialize parameters
    pis = jnp.ones(num_states)/num_states
    
    # Initialize kappas and mus
    key, key_1, key_2 = jr.split(key, 3)

    kappas = jnp.maximum(jr.normal(key_1, shape=(num_states, num_joints)) + 5, 0)

    rs = jnp.nansum(xs, axis=0)  # Shape (num_joints, dim)
    mus_mle = rs/jnp.linalg.norm(rs, axis=-1, keepdims=True)
    mus = VMF(mus_mle[None, :, :], jnp.ones((num_states,num_joints))*5).sample(seed=key_2)  # Sample with low concentration

    assert jnp.all(~jnp.isnan(xs))
    assert jnp.all(~jnp.isnan(mus))
    assert jnp.all(~jnp.isnan(kappas)) and jnp.all(kappas >= 0)

    init_carry = (jnp.empty((num_timesteps, num_states)),
                  pis, mus, kappas, jnp.empty(num_iters), xs)
    E_zs, pis, mus, kappas, lls, _ = lax.fori_loop(0, num_iters, em_step, init_carry)
    return lls, E_zs, pis, mus, kappas