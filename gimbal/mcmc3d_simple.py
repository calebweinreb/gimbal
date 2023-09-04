"""
gimbal/mcmc.py
"""
import jax.config
import jax.numpy as jnp
import jax.random as jr
from jax import lax, jit, vmap
from jax.scipy.special import logsumexp
from functools import partial

import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd

from .util import project
from .util import opencv_triangulate as triangulate


def initialize_parameters(params, pos_location_0=0., pos_variance_0=1e8):
    num_keypoints = len(params['parents'])
    num_cameras =  len(params['camera_matrices'])
    dim = params['camera_matrices'].shape[-1] - 1
    dim_obs = params['camera_matrices'].shape[-2] - 1

    dtype = jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32

    params['pos_dt_variance'] \
        = jnp.broadcast_to(params['pos_dt_variance'], (num_keypoints,))\
             .astype(dtype)
    params['pos_dt_covariance'] \
        = jnp.kron(jnp.diag(params['pos_dt_variance']), jnp.eye(dim))
    params['pos_dt_precision'] \
        = jnp.kron(jnp.diag(1./params['pos_dt_variance']), jnp.eye(dim))

    params['pos_variance_0'] \
        = jnp.broadcast_to(params.get('pos_variance_0', pos_variance_0),
                           (num_keypoints,))\
             .astype(dtype)
    params['pos_covariance_0'] \
        = jnp.kron(jnp.diag(params['pos_variance_0']), jnp.eye(dim))

    params['pos_location_0'] \
        = jnp.broadcast_to(params.get('pos_location_0', pos_location_0),
                           (num_keypoints, dim))\
             .astype(dtype)

    # -----------------------------
    # Observation error parameters
    # -----------------------------

    params['obs_outlier_location'] \
        = jnp.broadcast_to(params['obs_outlier_location'],
                           (num_cameras, num_keypoints, dim_obs))\
             .astype(dtype)
    params['obs_outlier_variance'] \
        = jnp.broadcast_to(params['obs_outlier_variance'],
                          (num_cameras, num_keypoints))\
             .astype(dtype)
    params['obs_outlier_covariance'] \
        = jnp.kron(params['obs_outlier_variance'][..., None, None],
                   jnp.eye(dim_obs))

    params['obs_inlier_location'] \
        = jnp.broadcast_to(params['obs_inlier_location'],
                          (num_cameras, num_keypoints, dim_obs))\
             .astype(dtype)
    params['obs_inlier_variance'] \
        = jnp.broadcast_to(params['obs_inlier_variance'],
                          (num_cameras, num_keypoints))\
             .astype(dtype)
    params['obs_inlier_covariance'] \
        = jnp.kron(params['obs_inlier_variance'][..., None, None],
                   jnp.eye(dim_obs))

    return params


@jit
def log_joint_probability(params, observations, outlier_prob,
                          outliers, positions):
    """Compute the log joint probabiltiy of sampled values under the
    prior parameters. 

    Parameters
    ----------
        params: dict
        observations: ndarray, shape (N, C, K, D_obs).
        outlier_prob: ndarray, shape (N, C, K).
        outliers: ndarray, shape (N, C, K).
        positions: ndarray, shape (N, K, D).
    """

    C = observations.shape[1]

    Z_prior = tfd.Bernoulli(probs=outlier_prob)
    
    Y_ins = [tfd.MultivariateNormalFullCovariance(
                        params['obs_inlier_location'][c],
                        params['obs_inlier_covariance'][c]) \
            for c in range(C)]
    Y_outs = [tfd.MultivariateNormalFullCovariance(
                        params['obs_outlier_location'][c],
                        params['obs_outlier_covariance'][c]) \
            for c in range(C)]
    X_t0 = tfd.MultivariateNormalFullCovariance(
                params['pos_location_0'].ravel(),
                params['pos_covariance_0'])

    # =================================================

    def log_likelihood(xt, yts, zts, params):
        """ Compute log likelihood of observations for a single time step.

        log p(y[t] | x[t], z[t]) = 
                z[t] log N(y[t] | proj(x[t]; P), omega_in)
                + (1-z[t]) log N(y[t] | proj(x[t]; P), omega_out)
        """
        lp = 0
        for c in range(C):
            obs_err = yts[c] - project(params["camera_matrices"][c], xt)
            lp += (1-zts[c]) * Y_ins[c].log_prob(obs_err)
            lp += zts[c] * Y_outs[c].log_prob(obs_err)
        return jnp.sum(lp)
    
    def log_pos_dynamics(xtp1, xt, params):
        """ Compute log p(x[t+1] | x[t]) = log N(x[t], sigma^2 I) """
        Xtp1_given_Xt = tfd.MultivariateNormalFullCovariance(
                                xt.ravel(), params['pos_dt_covariance'])
        return Xtp1_given_Xt.log_prob(xtp1.ravel())

    # =================================================
    
    # p(y | x, z)
    lp = jnp.sum(vmap(log_likelihood, in_axes=(0,0,0,None))
                     (positions, observations, outliers, params))
    
    # p(x[t] | x[t-1])
    lp += jnp.sum(vmap(log_pos_dynamics, in_axes=(0,0,None))
                      (positions[1:], positions[:-1], params))
    lp += X_t0.log_prob(positions[0].ravel())

    # p(z; rho)
    lp += jnp.sum(Z_prior.log_prob(outliers))

    return lp


def initialize(seed, params, observations, outlier_prob, 
               init_positions=None):
    """Initialize latent variables of model.

    Parameters
    ----------
        seed: jax.random.PRNGKey
        params: dict
        observations: ndarray, shape (N, C, K, D_obs)
        outlier_prob: ndarray, shape (N, C, K)
        init_positions: ndarray, shape (N, K, D), optional.
            Initial guess of 3D positions. If None (default), positions
            are triangulated using direct linear triangulation.
        
    Returns
    -------
        samples: dict
    """

    seed = iter(jr.split(seed, 3))
    N, C, K, D_obs = observations.shape
    
    # ---------------------
    # Initialize positions
    # ---------------------
    if init_positions is None:
        obs = jnp.moveaxis(observations, 1, 0)
        positions = triangulate(params['camera_matrices'], obs)
    else:
        positions = jnp.asarray(init_positions)

    # Initialize HMC results
    hmc_log_accept_ratio = jnp.array(0.)
    hmc_proposed_gradients = jnp.empty_like(positions)

    # ---------------------------
    # Sample outliers from prior
    # ---------------------------
    outliers = jr.uniform(next(seed), (N,C,K)) < outlier_prob

    # Consider any NaN observations as outliers
    outliers = jnp.where(jnp.isnan(observations).any(axis=-1),
                         True, outliers)

    log_prob = log_joint_probability(
                    params, observations, outlier_prob,
                    outliers, positions)

    return dict(
        outliers=outliers,
        positions=positions,
        log_probability=log_prob,
        hmc_log_accept_ratio=hmc_log_accept_ratio,
        hmc_proposed_gradients=hmc_proposed_gradients,
    )
  
@jit
def sample_positions(seed, params, observations, outlier_prob, samples):
    """Sample positions by taking one Hamiltonian Monte Carlo step."""
    
    def objective(positions):
        return log_joint_probability(
            params,
            observations,
            outlier_prob,
            samples['outliers'],
            positions )

    last_positions = samples['positions']   # shape (N, K, D)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=objective,
                num_leapfrog_steps=params['num_leapfrog_steps'],
                step_size=params['step_size']
            )

    positions, kernel_results = hmc.one_step(
                last_positions, hmc.bootstrap_results(last_positions), seed=seed)

    return positions, kernel_results

@jit
def sample_outliers(seed, params, observations, outlier_prob, samples):
    """Sample outliers
    
    TODO define inlier/outlier distributions beforehand. These are static.
    """

    predicted_observations = jnp.stack(
        [project(P, samples['positions']) for P in params['camera_matrices']],
        axis=1)
    error = observations - predicted_observations

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



def step(seed, params, observations, outlier_prob, samples):

    """Execute a single iteration of MCMC sampling."""
    seeds = jr.split(seed, 6)
    
    positions, kernel_results = sample_positions(
                    seeds[0], params, observations, outlier_prob, samples)
                    
    samples['positions'] = positions
    samples['hmc_log_accept_ratio'] = kernel_results.log_accept_ratio
    samples['hmc_proposed_gradients'] = kernel_results.proposed_results.grads_target_log_prob[0]
    samples['outliers'] = sample_outliers(seeds[1], params, observations, outlier_prob, samples)


    samples['log_probability'] = log_joint_probability(
                        params, observations, outlier_prob,
                        samples['outliers'], samples['positions'])
    return samples
