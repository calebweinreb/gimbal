"""
gimbal/mcmc2d.py
"""

import contextlib
import numpy as onp
from tqdm.auto import trange
import time

import jax.config
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jit, vmap
from jax.scipy.special import logsumexp
from functools import partial

import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd

from gimbal.util import (
    children_of, tree_graph_laplacian, 
    hvmfg_natural_parameter, cartesian_to_polar)


def vector_to_angle(V):
    y,x = V[...,1],V[...,0]+1e-10
    angles = (jnp.arctan(y/x)+(x<0)*jnp.pi)%(2*jnp.pi)-jnp.pi
    return angles

def gaussian_log_prob(x, mu, sigma_inv):
    return (-((mu-x)[...,None,:]*sigma_inv*(mu-x)[...,:,None]).sum((-1,-2))/2
            +jnp.log(jnp.linalg.det(sigma_inv))/2)-jnp.log(jnp.pi*2)*x.shape[-1]/2



def R_mat(h, D=2):
    """Create rotation matrices from an array of angles. If
    `D > 2` then rotation is performed in the first two dims 
    and the remaing axes are fixed.

    Parameters
    ----------
    h: ndarray, shape (*dims)
        Angles (in radians)

    Returns
    -------
    m: ndarray, shape (*dims, D, D)
        Stacked rotation matrices 
        
    D: int, default=2
        Dimension of each rotation matrix
    
    """
    m = jnp.tile(jnp.eye(D), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m


def initialize_parameters(
    params, pos_location_0=0., 
    pos_variance_0=1e8,
    state_transition_count=10., 
    regularizer=1e-6, dim=2):
    
    num_keypoints = len(params['parents'])
    num_states = len(params['state_probability'])

    # Must explicitly cast arrays (that are not results of calculations) to
    # desired dtype. For example, if we are in x64 mode and we have
    #   arr = jnp.array([1,2,3], dtype=jnp.float32)
    # the following operations preserve original dtype (undesirable)
    #   jnp.broadcast_to(arr, arr.shape) -> jnp.float32
    #   jnp.asarray(arr) -> jnp.float32
    #   jnp.array(arr) -> jnp.float32
    # The following operation changes dtype
    #   arr.asdtype(jnp.float64) -> jnp.float64
    dtype = jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32

    params['children'] = children_of(params['parents'])

    # -----------------------------------
    # Skeletal and positional parameters
    # -----------------------------------
    params['pos_radius'] \
        = jnp.broadcast_to(params['pos_radius'], (num_keypoints,))\
             .astype(dtype)

    params['pos_radial_variance'] \
        = jnp.broadcast_to(params['pos_radial_variance'], (num_keypoints,))\
             .astype(dtype)
    params['pos_radial_precision'] \
        = tree_graph_laplacian(
            params['parents'],
            1 / params['pos_radial_variance'][...,None,None] * jnp.eye(dim)
            ) + regularizer * jnp.eye(num_keypoints*dim)
    params['pos_radial_covariance'] \
        = jnp.linalg.inv(params['pos_radial_precision'])

    params['pos_dt_variance'] \
        = jnp.broadcast_to(params['pos_dt_variance'], (num_keypoints,))\
             .astype(dtype)

    params['pos_variance_0'] \
        = jnp.broadcast_to(params.get('pos_variance_0', pos_variance_0),
                           (num_keypoints,)).astype(dtype)

    params['pos_location_0'] \
        = jnp.broadcast_to(params.get('pos_location_0', pos_location_0),
                           (num_keypoints, dim)).astype(dtype)

    # -----------------------------
    # Observation error parameters
    # -----------------------------
    params['obs_outlier_location']  = jnp.broadcast_to(
        params['obs_outlier_location'], 
        (num_keypoints, dim)).astype(dtype)
    
    params['obs_outlier_variance'] = jnp.broadcast_to(
        params['obs_outlier_variance'], 
        (num_keypoints)).astype(dtype)
 
    params['obs_inlier_location'] = jnp.broadcast_to(
        params['obs_inlier_location'], 
        (num_keypoints, dim)).astype(dtype)
    
    params['obs_inlier_variance'] = jnp.broadcast_to(
        params['obs_inlier_variance'], 
        (num_keypoints)).astype(dtype)

    # -----------------------------
    # Pose state parameters
    # -----------------------------
    params['state_transition_count'] = jnp.broadcast_to(
        params.get('state_transition_count', state_transition_count), 
        (num_states,)).astype(dtype)
    
    params['state_probability'] = jnp.broadcast_to(
        params.get('state_probability', 1./num_states), 
        (num_states,)).astype(dtype)

    params['state_directions'] = jnp.broadcast_to(
        params.get('state_directions', jnp.array([0,0,1.])), 
        (num_states, num_keypoints, dim)).astype(dtype)

    params['state_concentrations'] = jnp.broadcast_to(
        params.get('state_concentrations', 0.),
        (num_states, num_keypoints)).astype(dtype)
    
    return params

#@jit
def log_joint_probability(params, observations,
                          outliers, positions, directions,
                          heading, pose_state, transition_matrix):
    """Compute the log joint probabiltiy of sampled values under the
    prior parameters. 

    Parameters
    ----------
        params: dict
        observations: ndarray, shape (N, K, D_obs).
        outliers: ndarray, shape (N, K).
        positions: ndarray, shape (N, K, D).
        directions: ndarray, shape (N, K, D).
        heading: ndarray, shape (N,).
        pose_state: ndarray, shape (N,).
        transition_matrix: ndarray, shape (S,S).
    """

    S = transition_matrix.shape[0]
    #Z_prior = tfd.Bernoulli(probs=params['obs_outlier_probability'])

    # =================================================

    def log_likelihood(xt, yt, zt, params):
        """ Compute log likelihood of observations for a single time step.

        log p(y[t] | x[t], z[t]) = 
                z[t] log N(y[t] | x[t], omega_in)
                + (1-z[t]) log N(y[t] | x[t], omega_out)
        """
        nan_mask = jnp.isnan(yt)
        lp = (1-zt) * vmap(gaussian_log_prob)(
            jnp.where(nan_mask, 0, yt), xt, jnp.eye(2)[None]/params['obs_inlier_variance'][...,None,None])
        lp += zt * vmap(gaussian_log_prob)(
            jnp.where(nan_mask, 0, yt), xt, jnp.eye(2)[None]/params['obs_outlier_variance'][...,None,None])
        return jnp.where(nan_mask.any(-1), 0, lp).sum()
    
    def log_pos_given_dir(xt, ut, params):
        """ Compute log likelihood of positions given directions for a
        single time step.

        log p(x[t] | u[t]) = prod_j N(x[t,parj] + r[j] u[t,j], sigma^2 I)
        """
        ht = hvmfg_natural_parameter(
            params['children'], params['pos_radius'],
            params['pos_radial_variance'],ut)
        
        Xt_given_Ut = tfd.MultivariateNormalFullCovariance(
            params['pos_radial_covariance'] @ ht.ravel(),
            params['pos_radial_covariance'])
        
        return Xt_given_Ut.log_prob(xt.ravel())
    
    
    def log_pos_dynamics(xtp1, xt, params):
        """ Compute log p(x[t+1] | x[t]) = log N(x[t], sigma^2 I) """
        return vmap(gaussian_log_prob)(xt, xtp1, jnp.eye(2)[None]/params['pos_dt_variance'][...,None,None]).sum()
    
    def log_dir_given_state(ut, ht, st, params):
        """ Compute log p(R(-h[t]) u[t] | h[t], s[t]) = log vMF(nu_s[t], kappa_s[t]) """
        canon_ut = ut @ R_mat(-ht).T
        u_given_st = tfd.VonMisesFisher(
            params['state_directions'][st],
            params['state_concentrations'][st])
        return jnp.sum(u_given_st.log_prob(canon_ut))

    # =================================================
    
    # p(y | x, z)
    lp = jnp.sum(vmap(log_likelihood, in_axes=(0,0,0,None))(positions, observations, outliers, params))

    # p(z; rho)
    #lp += jnp.sum(Z_prior.log_prob(outliers))

    # p(x | u)
    lp += jnp.sum(vmap(log_pos_given_dir, in_axes=(0,0,None))
                      (positions, directions, params))

    # p(x[t] | x[t-1])
    lp += jnp.sum(vmap(log_pos_dynamics, in_axes=(0,0,None))
                      (positions[1:], positions[:-1], params))
    # p(u | s)
    lp += jnp.sum(vmap(log_dir_given_state, in_axes=(0,0,0,None))
                      (directions, heading, pose_state, params))

    # p(h) = VonMises(0, 0), so lp = max entropy over circle, which is constant

    # p(s[0]) + sum p(s[t+1] | s[t])
    lp += jnp.sum(jnp.log(transition_matrix[pose_state[1:], pose_state[:-1]]))
    lp += jnp.sum(jnp.log(params['state_probability']))

    return lp

def hack_interp(observations, outliers):
    x = jnp.arange(observations.shape[0])
    xp = jnp.where(outliers==0, x, -1e6)
    fp = jnp.where(outliers==0, observations, 0)
    o = jnp.argsort(xp)
    return jnp.interp(x, xp[o], fp[o])


def initialize(seed, params, observations, outlier_prob):
    """Initialize latent variables of model.

    Parameters
    ----------
        seed: jax.random.PRNGKey
        params: dict
        observations: ndarray, shape (N, K, D)

    Returns
    -------
        samples: dict
    """

    seed = iter(jr.split(seed, 3))
    N, K, D_obs = observations.shape
    
    # ---------------------------
    # Sample outliers from prior
    # ---------------------------
    outliers = jr.uniform(next(seed), (N,K)) < outlier_prob


    # ----------------------------------------------
    # Initialize positions by interpolating outliers
    # ----------------------------------------------
    interp = jax.vmap(hack_interp, in_axes=1, out_axes=1)
    positions = jnp.stack([
        interp(observations[:,:,0], outliers),
        interp(observations[:,:,1], outliers)], axis=2)
    

    # Initialize HMC results
    hmc_log_accept_ratio = jnp.array(0.)
    hmc_proposed_gradients = jnp.empty_like(positions)

    # --------------------------
    # Derive initial directions
    # --------------------------
    directions = positions - positions[:,params['parents']]
    directions /= jnp.linalg.norm(directions, axis=-1, keepdims=True)

    # Use placeholder value for undefined root direction
    directions = jnp.nan_to_num(directions.at[:,0].set(0.))

    # -------------------------------
    # Derive initial heading vectors
    # -------------------------------
    k_base, k_tip = params['crf_keypoints']
    
    # Unnormalized direction vector
    heading = positions[:,k_tip] - positions[:,k_base]
    heading /= jnp.maximum(jnp.linalg.norm(heading, axis=-1, keepdims=True),1e-6)
    heading = vector_to_angle(heading[...,:2])
    

    # ------------------------------------------------------
    # Sample pose states and transition matrix from uniform
    # ------------------------------------------------------
    # TODO sample from prior parameters, state_probability and state_counts
    num_states = len(params['state_probability'])
    pose_state = jr.randint(next(seed), (N,), 0, num_states)
    transition_matrix = jnp.ones((num_states, num_states)) / num_states

    log_prob = log_joint_probability(
                    params, observations,
                    outliers, positions, directions,
                    heading, pose_state, transition_matrix,
                    )

    return dict(
        outliers=outliers,
        positions=positions,
        directions=directions,
        heading=heading,
        pose_state=pose_state,
        transition_matrix=transition_matrix,
        log_probability=log_prob,
        hmc_log_accept_ratio=hmc_log_accept_ratio,
        hmc_proposed_gradients=hmc_proposed_gradients,
    )
  
@jit
def sample_positions(seed, params, observations, samples):
    """Sample positions by taking one Hamiltonian Monte Carlo step."""
    
    N, K, D_obs = observations.shape
    
    def objective(positions):
        return log_joint_probability(
            params,
            observations,
            samples['outliers'],
            positions,
            samples['directions'],
            samples['heading'],
            samples['pose_state'],
            samples['transition_matrix'],
            )

    last_positions = samples['positions']   # shape (N, K, D)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=objective,
                num_leapfrog_steps=params['num_leapfrog_steps'],
                step_size=params['step_size']
            )

    positions, kernel_results = hmc.one_step(
                last_positions, 
                hmc.bootstrap_results(last_positions),
                seed=seed)

    return positions, kernel_results

@jit
def sample_outliers(seed, params, observations, samples, outlier_prob):
    """Sample outliers
    
    TODO define inlier/outlier distributions beforehand. These are static.
    """
    error = observations - samples['positions']

    # Log probability of predicted observation being inlier
    Y_inlier = tfd.MultivariateNormalFullCovariance(
                                    params['obs_inlier_location'], 
                                    params['obs_inlier_covariance']
                                    )
    lp_k0 = jnp.log(1-outlier_prob)
    lp_k0 += Y_inlier.log_prob(error)

    # Log probability of predicted observation being outlier
    Y_outlier = tfd.MultivariateNormalFullCovariance(
                                    params['obs_outlier_location'], 
                                    params['obs_outlier_covariance']
                                    )
    lp_k1 = jnp.log(outlier_prob)
    lp_k1 += Y_outlier.log_prob(error)
    
    # Update posterior
    lognorm = logsumexp(jnp.stack([lp_k0, lp_k1], axis=-1), axis=-1)
    p_isoutlier = jnp.exp(lp_k1 - lognorm)
    
    # Draw new samples
    outliers = jr.uniform(seed, observations.shape[:-1]) < p_isoutlier

    # Any NaN observations are obviously drawn from outlier distribution
    outliers = jnp.where(jnp.isnan(observations[...,0]), True, outliers)

    return outliers

@jit
def sample_directions(seed, params, samples):
    """Sample directions from 3D von Mises-Fisher distribution."""

    positions = samples['positions']    # (N,K,3)
    heading = samples['heading']        # (N,)
    state = samples['pose_state']       # (N,)
    D = positions.shape[-1]

    # Rotate mean directional prior (defined in the canonical reference 
    # frame) into the absolute/ambient reference frame
    mu_priors = jnp.einsum('tmn, tjn-> tjm',
                           R_mat(heading), params['state_directions'][state])
    kappa_priors = params['state_concentrations'][state]

    # Keypoint position contributions to direction vectors
    dx = positions - positions[:, params['parents']]
    
    # Calculate posterior parameters
    mu_tilde  = kappa_priors[...,None] * mu_priors
    mu_tilde += dx * (params['pos_radius']/params['pos_radial_variance'])[:,None]
    
    mu_post = mu_tilde / jnp.linalg.norm(mu_tilde, axis=-1, keepdims=True)
    kappa_post = jnp.linalg.norm(mu_tilde, axis=-1)
    
    # Sample from posterior
    directions = tfd.VonMisesFisher(mu_post, kappa_post).sample(seed=seed)
    directions = directions.at[:,0,:].set(jnp.zeros(D))

    return directions

@jit
def sample_headings(seed, params, samples):
    """Sample headings from uniform circular distribution."""

    state = samples['pose_state']   # (N,)

    # Polar representation of 3D vectors, array shapes (N,K)
    mu_thetas, mu_phis = cartesian_to_polar(params['state_directions'][state][:,1:])
    us_thetas, us_phis = cartesian_to_polar(samples['directions'][:,1:])

    # Update parameters, which we find by solving
    #   k_likelihood sin(theta_likelihood) = sum w_i sin(theta_i)
    #   k_likelihood cos(theta_likelihood) = sum w_i cos(theta_i)
    # Recall: Our prior for headings is vMF with concentration 0
    # so, theta_post = theta_likelihood, k_post = k_likelihood
    azim_weights = jnp.sin(mu_phis) * jnp.sin(us_phis)
    k_sin_thetas = jnp.sum(azim_weights * jnp.sin(us_thetas - mu_thetas), axis=-1)
    k_cos_thetas = jnp.sum(azim_weights * jnp.cos(us_thetas - mu_thetas), axis=-1)

    theta_post = jnp.arctan2(k_sin_thetas, k_cos_thetas)
    kappa_post = jnp.sqrt(k_sin_thetas**2 + k_cos_thetas**2)
    
    return tfd.VonMises(theta_post, kappa_post).sample(seed=seed)

@jit
def U_given_S_log_likelihoods(params, samples):
    """Calculate log likelihood of sampled directions under conditional
    prior distribution.

    Function seperated from sample_state so that this portion can be jitted.
    """
    directions = samples['directions']  # (N, K, 2)
    heading = samples['heading']        # (N,)

    # Rotate direction in absolute frame to canonical reference frame
    canonical_directions = jnp.einsum('tmn, tjn-> tjm',
                                      R_mat(-heading), directions)

    # Calculate log likelihood of each set of directions, for each state
    # TODO Initialize
    # shape: (N, S)
    U_given_S = tfd.VonMisesFisher(params['state_directions'],
                                   params['state_concentrations'])
    return jnp.sum(U_given_S.log_prob(canonical_directions[:,None,...]),
                   axis=-1)



def sample_state(seed, params, samples):
    lls = U_given_S_log_likelihoods(params, samples)
    lls += jnp.log(params['state_probability']+1e-16)
    return jr.categorical(seed, lls)

@jit
def step(seed, params, samples, observations, outlier_prob):

    """Execute a single iteration of MCMC sampling."""
    seeds = jr.split(seed, 6)
    
    samples['outliers'] = sample_outliers(
        seeds[0], params, observations, samples, outlier_prob)
    
    positions, kernel_results = sample_positions(
        seeds[1], params, observations, samples)
      
    samples['positions'] = positions
    samples['hmc_log_accept_ratio'] = kernel_results.log_accept_ratio
    samples['hmc_proposed_gradients'] = kernel_results.proposed_results.grads_target_log_prob[0]
 
    samples['directions'] = sample_directions(
        seeds[2], params, samples)
  
    samples['heading'] = sample_headings(
        seeds[3], params, samples)
  
    samples['pose_state'] = sample_state(
        seeds[4], params, samples)

    samples['log_probability'] = log_joint_probability(
        params, observations, samples['outliers'],
        samples['positions'], samples['directions'],
        samples['heading'], samples['pose_state'],
        samples['transition_matrix'])
  
    return samples

