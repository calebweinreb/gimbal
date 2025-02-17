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
from dynamax.hidden_markov_model import parallel_hmm_posterior_sample

from .util import (
    project,
    xyz_to_uv,
    uv_to_xyz,
    signed_angular_difference,
    Rxy_mat,
    cartesian_to_polar,
    children_of,
    tree_graph_laplacian,
    hvmfg_natural_parameter,
)
from .util import opencv_triangulate as triangulate
from . import util_io


def initialize_parameters(
    params,
    pos_location_0=0.0,
    pos_variance_0=1e8,
    state_transition_count=10.0,
    regularizer=1e-6,
):
    num_keypoints = len(params["parents"])
    num_cameras = len(params["camera_matrices"])
    dim = params["camera_matrices"].shape[-1] - 1
    dim_obs = params["camera_matrices"].shape[-2] - 1
    num_states = len(params["state_probability"])

    # Must explicitly cast arrays (that are not results of calculations) to
    # desired dtype. For example, if we are in x64 mode and we have
    #   arr = jnp.array([1,2,3], dtype=jnp.float32)
    # the following operations preserve original dtype (undesirable)
    #   jnp.broadcast_to(arr, arr.shape) -> jnp.float32
    #   jnp.asarray(arr) -> jnp.float32
    #   jnp.array(arr) -> jnp.float32
    # The following operation changes dtype
    #   arr.asdtype(jnp.float64) -> jnp.float64
    dtype = jnp.float64 if jax.config.read("jax_enable_x64") else jnp.float32

    params["children"] = children_of(params["parents"])

    # -----------------------------------
    # Canonical reference frame
    # -----------------------------------
    x_axis, z_axis = params["crf_abscissa"], params["crf_normal"]
    y_axis = jnp.cross(z_axis, x_axis)
    params["crf_axes"] = jnp.stack([x_axis, y_axis, z_axis], axis=0)

    # -----------------------------------
    # Skeletal and positional parameters
    # -----------------------------------
    params["pos_radius"] = jnp.broadcast_to(
        params["pos_radius"], (num_keypoints,)
    ).astype(dtype)

    params["pos_radial_variance"] = jnp.broadcast_to(
        params["pos_radial_variance"], (num_keypoints,)
    ).astype(dtype)
    params["pos_radial_precision"] = tree_graph_laplacian(
        params["parents"],
        1 / params["pos_radial_variance"][..., None, None] * jnp.eye(dim),
    ) + regularizer * jnp.eye(num_keypoints * dim)
    params["pos_radial_covariance"] = jnp.linalg.inv(params["pos_radial_precision"])

    # NB: Using jnp.linalg.eigvals throws the NotImplementedError with message "Nonsymmetric eigendecomposition is only implemented on the CPU backend. However, these matrices ARE symmetric...
    # assert jnp.all(onp.linalg.eigvals(params['pos_radial_precision']) > 0.), \
    #     f"Expected positive definite radial precision matrix, but got eigenvalues of {jnp.linalg.eigvals(params['pos_radial_precision'])}.\nConsider adjusting value of regularizer."
    # assert jnp.all(onp.linalg.eigvals(params['pos_radial_covariance']) > 0.), \
    #     f"Expected positive definite radial covariance matrix, but got eigenvalues of {jnp.linalg.eigvals(params['pos_radial_covariance'])}.\nConsider adjusting value of regularizer."

    params["pos_dt_variance"] = jnp.broadcast_to(
        params["pos_dt_variance"], (num_keypoints,)
    ).astype(dtype)
    params["pos_dt_covariance"] = jnp.kron(
        jnp.diag(params["pos_dt_variance"]), jnp.eye(dim)
    )
    params["pos_dt_precision"] = jnp.kron(
        jnp.diag(1.0 / params["pos_dt_variance"]), jnp.eye(dim)
    )

    params["pos_precision_t"] = (
        params["pos_radial_precision"] + params["pos_dt_precision"]
    )
    params["pos_covariance_t"] = jnp.linalg.inv(params["pos_precision_t"])

    params["pos_variance_0"] = jnp.broadcast_to(
        params.get("pos_variance_0", pos_variance_0), (num_keypoints,)
    ).astype(dtype)
    params["pos_covariance_0"] = jnp.kron(
        jnp.diag(params["pos_variance_0"]), jnp.eye(dim)
    )

    params["pos_location_0"] = jnp.broadcast_to(
        params.get("pos_location_0", pos_location_0), (num_keypoints, dim)
    ).astype(dtype)

    # -----------------------------
    # Observation error parameters
    # -----------------------------
    params["obs_outlier_location"] = jnp.broadcast_to(
        params["obs_outlier_location"], (num_cameras, num_keypoints, dim_obs)
    ).astype(dtype)
    params["obs_outlier_variance"] = jnp.broadcast_to(
        params["obs_outlier_variance"], (num_cameras, num_keypoints)
    ).astype(dtype)
    params["obs_outlier_covariance"] = jnp.kron(
        params["obs_outlier_variance"][..., None, None], jnp.eye(dim_obs)
    )

    params["obs_inlier_location"] = jnp.broadcast_to(
        params["obs_inlier_location"], (num_cameras, num_keypoints, dim_obs)
    ).astype(dtype)
    params["obs_inlier_variance"] = jnp.broadcast_to(
        params["obs_inlier_variance"], (num_cameras, num_keypoints)
    ).astype(dtype)
    params["obs_inlier_covariance"] = jnp.kron(
        params["obs_inlier_variance"][..., None, None], jnp.eye(dim_obs)
    )

    # -----------------------------
    # Pose state parameters
    # -----------------------------
    params["state_transition_count"] = jnp.broadcast_to(
        params.get("state_transition_count", state_transition_count), (num_states,)
    ).astype(dtype)

    params["state_probability"] = jnp.broadcast_to(
        params.get("state_probability", 1.0 / num_states), (num_states,)
    ).astype(dtype)

    params["state_directions"] = jnp.broadcast_to(
        params.get("state_directions", jnp.array([0, 0, 1.0])),
        (num_states, num_keypoints, dim),
    ).astype(dtype)

    params["state_concentrations"] = jnp.broadcast_to(
        params.get("state_concentrations", 0.0), (num_states, num_keypoints)
    ).astype(dtype)

    return params


@partial(jit, static_argnames=("ignore_anatomy",))
def log_joint_probability(
    params,
    observations,
    outlier_prob,
    outliers,
    positions,
    directions,
    heading,
    pose_state,
    transition_matrix,
    ignore_anatomy=False,
):
    """Compute the log joint probabiltiy of sampled values under the
    prior parameters.

    Parameters
    ----------
        params: dict
        observations: ndarray, shape (N, C, K, D_obs).
        outlier_prob: ndarray, shape (N, C, K).
        outliers: ndarray, shape (N, C, K).
        positions: ndarray, shape (N, K, D).
        directions: ndarray, shape (N, K, D).
        heading: ndarray, shape (N,).
        pose_state: ndarray, shape (N,).
        transition_matrix: ndarray, shape (S,S).
        ignore_anatomy: bool, optional. default: False
            If True, only include positions and outliers in the model.
    """

    C = observations.shape[1]

    Z_prior = tfd.Bernoulli(probs=outlier_prob)

    Y_ins = [
        tfd.MultivariateNormalFullCovariance(
            params["obs_inlier_location"][c], params["obs_inlier_covariance"][c]
        )
        for c in range(C)
    ]
    Y_outs = [
        tfd.MultivariateNormalFullCovariance(
            params["obs_outlier_location"][c], params["obs_outlier_covariance"][c]
        )
        for c in range(C)
    ]
    X_t0 = tfd.MultivariateNormalFullCovariance(
        params["pos_location_0"].ravel(), params["pos_covariance_0"]
    )
    # U_given_S = VMF(params['pstate']['mus'], params['pstate']['kappas'])  # batch shape (num_states, num_joints), event_shape (dim)

    # =================================================

    def log_likelihood(xt, yts, zts, params):
        """Compute log likelihood of observations for a single time step.

        log p(y[t] | x[t], z[t]) =
                z[t] log N(y[t] | proj(x[t]; P), omega_in)
                + (1-z[t]) log N(y[t] | proj(x[t]; P), omega_out)
        """

        # Does not work :( ... Error message
        # Shapes must be 1D sequences of concrete values of integer type, got [Traced<ShapedArray(int32[])>with<DynamicJaxprTrace(level=0/2)>].
        # If using `jit`, try using `static_argnums` or applying `jit` to smaller subfunctions.
        # obs_err = np.stack([yts[c] - project(xt, params["Ps"][c]) \
        #                     for c in range(yts.shape[0])], axis=0)
        # lp = (1-zts) * Y_ins.log_prob(obs_err) + zts * Y_outs.log_prob(obs_err)

        lp = 0
        # for c in range(yts.shape[0]):
        for c in range(C):
            obs_err = yts[c] - project(params["camera_matrices"][c], xt)
            lp += (1 - zts[c]) * Y_ins[c].log_prob(obs_err)
            lp += zts[c] * Y_outs[c].log_prob(obs_err)
            # Y_in = tfd.MultivariateNormalFullCovariance(params['obs_inlier_location'][c], params['obs_inlier_covariance'][c])
            # Y_out = tfd.MultivariateNormalFullCovariance(params['obs_outlier_location'][c], params['obs_outlier_covariance'][c])
            # lp += (1-zts[c]) * Y_in.log_prob(obs_err)
            # lp += zts[c] * Y_out.log_prob(obs_err)
        return jnp.sum(lp)

    def log_pos_given_dir(xt, ut, params):
        """Compute log likelihood of positions given directions for a
        single time step.

        log p(x[t] | u[t]) = prod_j N(x[t,parj] + r[j] u[t,j], sigma^2 I)
        """
        ht = hvmfg_natural_parameter(
            params["children"], params["pos_radius"], params["pos_radial_variance"], ut
        )
        Xt_given_Ut = tfd.MultivariateNormalFullCovariance(
            params["pos_radial_covariance"] @ ht.ravel(),
            params["pos_radial_covariance"],
        )
        return Xt_given_Ut.log_prob(xt.ravel())

    def log_pos_dynamics(xtp1, xt, params):
        """Compute log p(x[t+1] | x[t]) = log N(x[t], sigma^2 I)"""
        Xtp1_given_Xt = tfd.MultivariateNormalFullCovariance(
            xt.ravel(), params["pos_dt_covariance"]
        )
        return Xtp1_given_Xt.log_prob(xtp1.ravel())

    def log_dir_given_state(ut, ht, st, params):
        """Compute log p(R(-h[t]) u[t] | h[t], s[t]) = log vMF(nu_s[t], kappa_s[t])"""
        canon_ut = ut @ Rxy_mat(-ht).T
        u_given_st = tfd.VonMisesFisher(
            params["state_directions"][st], params["state_concentrations"][st]
        )
        return jnp.sum(u_given_st.log_prob(canon_ut))

    # =================================================

    # p(y | x, z)
    lp = jnp.sum(
        vmap(log_likelihood, in_axes=(0, 0, 0, None))(
            positions, observations, outliers, params
        )
    )

    # p(x[t] | x[t-1])
    lp += jnp.sum(
        vmap(log_pos_dynamics, in_axes=(0, 0, None))(
            positions[1:], positions[:-1], params
        )
    )
    lp += X_t0.log_prob(positions[0].ravel())

    # p(z; rho)
    lp += jnp.sum(Z_prior.log_prob(outliers))

    if not ignore_anatomy:

        # p(x | u)
        lp += jnp.sum(
            vmap(log_pos_given_dir, in_axes=(0, 0, None))(positions, directions, params)
        )

        # p(u | s)
        # canon_u = jnp.einsum('tmn, tjn-> tjm', R_mat(-h), u)
        # lp = jnp.sum(U_given_S[s].log_prob(canon_u))
        lp += jnp.sum(
            vmap(log_dir_given_state, in_axes=(0, 0, 0, None))(
                directions, heading, pose_state, params
            )
        )

        # p(h) = VonMises(0, 0), so lp = max entropy over circle, which is constant

        # p(s[0]) + sum p(s[t+1] | s[t])
        lp += jnp.sum(jnp.log(transition_matrix[pose_state[1:], pose_state[:-1]]))
        lp += jnp.sum(jnp.log(params["state_probability"]))

    return lp


def initialize(
    seed, params, observations, outlier_prob, init_positions=None, ignore_anatomy=False
):
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
        ignore_anatomy: bool, optional. default: False
            If True, only include positions and outliers in the model.

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
        if D_obs == 2:
            obs = jnp.moveaxis(observations, 1, 0)
            positions = triangulate(params["camera_matrices"], obs)
        elif D_obs == 3:
            positions = observations.copy()
        else:
            raise ValueError(
                f"Expected observation data dimension {{2,3}}, but receieved {D_obs}."
            )
    else:
        positions = jnp.asarray(init_positions)

    # Initialize HMC results
    hmc_log_accept_ratio = jnp.array(0.0)
    hmc_proposed_gradients = jnp.empty_like(positions)

    # ---------------------------
    # Sample outliers from prior
    # ---------------------------
    outliers = jr.uniform(next(seed), (N, C, K)) < outlier_prob

    # Consider any NaN observations as outliers
    outliers = jnp.where(jnp.isnan(observations).any(axis=-1), True, outliers)

    if not ignore_anatomy:
        # --------------------------
        # Derive initial directions
        # --------------------------
        directions = positions - positions[:, params["parents"]]
        directions /= jnp.linalg.norm(directions, axis=-1, keepdims=True)

        # Use placeholder value for undefined root direction
        directions = directions.at[:, 0].set(0.0)

        # -------------------------------
        # Derive initial heading vectors
        # -------------------------------
        k_base, k_tip = params["crf_keypoints"]
        u_axis, v_axis, n_axis = params["crf_axes"]

        # Unnormalized direction vector
        heading = positions[:, k_tip] - positions[:, k_base]

        # Project 3D direction vectors onto 2D plane of coordinate reference
        # frame (CRF). Then, re-embed the 2D plane in the ambient 3D frame.
        # In the default case, the CRF is defined as the standard Cartesian
        # reference (i.e. poses aligned to "x"-axis, and headings considered
        # in the "x-y" plane, rotating about normal "z"-axis.) The following
        # two operations then effectively zero-out the z-axis coordinates.
        heading = xyz_to_uv(heading, u_axis, v_axis)
        heading = uv_to_xyz(heading, u_axis, v_axis)

        # Normalize vector to unit length
        heading /= jnp.linalg.norm(heading, axis=-1, keepdims=True)

        # Calculate angle from canonical reference direction
        heading = signed_angular_difference(
            jnp.broadcast_to(u_axis, heading.shape), heading, n_axis
        )

        # ------------------------------------------------------
        # Sample pose states and transition matrix from uniform
        # ------------------------------------------------------
        # TODO sample from prior parameters, state_probability and state_counts
        num_states = len(params["state_probability"])
        pose_state = jr.randint(next(seed), (N,), 0, num_states)
        transition_matrix = jnp.ones((num_states, num_states)) / num_states

    else:
        heading = None
        directions = None
        pose_state = None
        transition_matrix = None

    log_prob = log_joint_probability(
        params,
        observations,
        outlier_prob,
        outliers,
        positions,
        directions,
        heading,
        pose_state,
        transition_matrix,
        ignore_anatomy=ignore_anatomy,
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


@partial(jit, static_argnames=("ignore_anatomy",))
def sample_positions(
    seed, params, observations, outlier_prob, samples, ignore_anatomy=False
):
    """Sample positions by taking one Hamiltonian Monte Carlo step."""

    N, C, K, D_obs = observations.shape

    def objective(positions):
        return log_joint_probability(
            params,
            observations,
            outlier_prob,
            samples["outliers"],
            positions,
            samples["directions"],
            samples["heading"],
            samples["pose_state"],
            samples["transition_matrix"],
            ignore_anatomy=ignore_anatomy,
        )

    last_positions = samples["positions"]  # shape (N, K, D)

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=objective,
        num_leapfrog_steps=params["num_leapfrog_steps"],
        step_size=params["step_size"],
    )

    positions, kernel_results = hmc.one_step(
        last_positions, hmc.bootstrap_results(last_positions), seed=seed
    )

    return positions, kernel_results


@jit
def sample_outliers(seed, params, observations, outlier_prob, samples):
    """Sample outliers

    TODO define inlier/outlier distributions beforehand. These are static.
    """

    predicted_observations = jnp.stack(
        [project(P, samples["positions"]) for P in params["camera_matrices"]], axis=1
    )
    error = observations - predicted_observations

    # Log probability of predicted observation being inlier
    Y_inlier = tfd.MultivariateNormalFullCovariance(
        params["obs_inlier_location"], params["obs_inlier_covariance"]
    )
    lp_k0 = jnp.log(1 - outlier_prob)
    lp_k0 += Y_inlier.log_prob(error)

    # Log probability of predicted observation being outlier
    Y_outlier = tfd.MultivariateNormalFullCovariance(
        params["obs_outlier_location"], params["obs_outlier_covariance"]
    )
    lp_k1 = jnp.log(outlier_prob)
    lp_k1 += Y_outlier.log_prob(error)

    # Update posterior
    lognorm = logsumexp(jnp.stack([lp_k0, lp_k1], axis=-1), axis=-1)
    p_isoutlier = jnp.exp(lp_k1 - lognorm)

    # Draw new samples
    outliers = jr.uniform(seed, observations.shape[:-1]) < p_isoutlier

    # Any NaN observations are obviously drawn from outlier distribution
    outliers = jnp.where(jnp.isnan(observations[..., 0]), True, outliers)

    return outliers


@jit
def sample_directions(seed, params, samples):
    """Sample directions from 3D von Mises-Fisher distribution."""

    positions = samples["positions"]  # (N,K,3)
    heading = samples["heading"]  # (N,)
    state = samples["pose_state"]  # (N,)
    D = positions.shape[-1]

    # Rotate mean directional prior (defined in the canonical reference
    # frame) into the absolute/ambient reference frame
    mu_priors = jnp.einsum(
        "tmn, tjn-> tjm", Rxy_mat(heading), params["state_directions"][state]
    )
    kappa_priors = params["state_concentrations"][state]

    # Keypoint position contributions to direction vectors
    dx = positions - positions[:, params["parents"]]

    # Calculate posterior parameters
    mu_tilde = kappa_priors[..., None] * mu_priors
    mu_tilde += dx * (params["pos_radius"] / params["pos_radial_variance"])[:, None]

    mu_post = mu_tilde / jnp.linalg.norm(mu_tilde, axis=-1, keepdims=True)
    kappa_post = jnp.linalg.norm(mu_tilde, axis=-1)

    # Sample from posterior
    directions = tfd.VonMisesFisher(mu_post, kappa_post).sample(seed=seed)
    directions = directions.at[:, 0, :].set(jnp.zeros(D))

    return directions


@jit
def sample_headings(seed, params, samples):
    """Sample headings from uniform circular distribution."""
    state = samples["pose_state"]  # (N,)

    # Polar representation of 3D vectors, array shapes (N,K)
    mu_thetas, mu_phis = cartesian_to_polar(params["state_directions"][state][:, 1:])
    us_thetas, us_phis = cartesian_to_polar(samples["directions"][:, 1:])

    # Update parameters, which we find by solving
    #   k_likelihood sin(theta_likelihood) = sum w_i sin(theta_i)
    #   k_likelihood cos(theta_likelihood) = sum w_i cos(theta_i)
    # Recall: Our prior for headings is vMF with concentration 0
    # so, theta_post = theta_likelihood, k_post = k_likelihood
    azim_weights = jnp.sin(mu_phis) * jnp.sin(us_phis)
    k_sin_thetas = jnp.sum(azim_weights * jnp.sin(us_thetas - mu_thetas), axis=-1)
    k_cos_thetas = jnp.sum(azim_weights * jnp.cos(us_thetas - mu_thetas), axis=-1)

    # Use VonMisesFisher instead of VonMises to avoid NaNs
    kappa_post = jnp.sqrt(k_sin_thetas**2 + k_cos_thetas**2)
    direction_post = jnp.stack([k_cos_thetas, k_sin_thetas], axis=-1)
    direction_post /= kappa_post[..., None]
    sampled_dir = tfd.VonMisesFisher(direction_post, kappa_post).sample(seed=seed)
    return jnp.arctan2(sampled_dir[:, 1], sampled_dir[:, 0])


@jit
def U_given_S_log_likelihoods(params, samples):
    """Calculate log likelihood of sampled directions under conditional
    prior distribution.

    Function seperated from sample_state so that this portion can be jitted.
    """
    directions = samples["directions"]  # (N, K, 3)
    heading = samples["heading"]  # (N,)

    # Rotate direction in absolute frame to canonical reference frame
    canonical_directions = jnp.einsum("tmn, tjn-> tjm", Rxy_mat(-heading), directions)

    # Calculate log likelihood of each set of directions, for each state
    # TODO Initialize
    # shape: (N, S)
    U_given_S = tfd.VonMisesFisher(
        params["state_directions"], params["state_concentrations"]
    )
    return jnp.sum(U_given_S.log_prob(canonical_directions[:, None, ...]), axis=-1)


def sample_state(seed, params, samples):
    transition_matrix = samples["transition_matrix"]  # shape (S, S)
    initial_distribution = params["state_probability"]  # shape (S,)
    lls = U_given_S_log_likelihoods(params, samples)
    return parallel_hmm_posterior_sample(
        seed, initial_distribution, transition_matrix, lls
    )[1]


# def sample_state(seed, params, samples):
#     lls = U_given_S_log_likelihoods(params, samples)
#     lls += jnp.log(params['state_probability']+1e-16)
#     return jr.categorical(seed, lls)


@jit
def sample_transition_matrix(seed, params, samples):
    state = samples["pose_state"]  # shape (N,)
    N, S = len(state), len(params["state_transition_count"])

    def count_all(t, counts):
        return counts.at[state[t], state[t + 1]].add(1)

    counts = lax.fori_loop(0, N - 1, count_all, jnp.zeros((S, S)))
    return tfd.Dirichlet(params["state_transition_count"] + counts).sample(seed=seed)


def step(seed, params, observations, outlier_prob, samples, ignore_anatomy=False):
    """Execute a single iteration of MCMC sampling."""
    seeds = jr.split(seed, 6)

    positions, kernel_results = sample_positions(
        seeds[0],
        params,
        observations,
        outlier_prob,
        samples,
        ignore_anatomy=ignore_anatomy,
    )

    samples["positions"] = positions
    samples["hmc_log_accept_ratio"] = kernel_results.log_accept_ratio
    samples["hmc_proposed_gradients"] = (
        kernel_results.proposed_results.grads_target_log_prob[0]
    )
    samples["outliers"] = sample_outliers(
        seeds[1], params, observations, outlier_prob, samples
    )

    if not ignore_anatomy:
        samples["directions"] = sample_directions(seeds[2], params, samples)
        samples["heading"] = sample_headings(seeds[3], params, samples)
        samples["pose_state"] = sample_state(seeds[4], params, samples)
        samples["transition_matrix"] = sample_transition_matrix(
            seeds[5], params, samples
        )

    samples["log_probability"] = log_joint_probability(
        params,
        observations,
        outlier_prob,
        samples["outliers"],
        samples["positions"],
        samples["directions"],
        samples["heading"],
        samples["pose_state"],
        samples["transition_matrix"],
        ignore_anatomy=ignore_anatomy,
    )
    return samples
