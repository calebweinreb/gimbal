import jax.numpy as jnp
import jax.scipy.special
import jax.random as jr
import numpy as onp

from tensorflow_probability.substrates import jax as tfp

from .util import (log_bessel_iv_asymptotic, log_sinh, coth, children_of)

CONCENTRATION_REGULARIZER = 1e-8
VARIANCE_REGULARIZER = 1e-8

def vmf_mean_2d(mean_direction, concentration):
    """Calculate the mean of a 2D vMF distribution.
    
    The mean is given by

    .. math:
        \mathbb{E}[p(u)] = \frac{I_1(\kappa)}{I_0(\kappa)} \nu

    where :math:`I_j` denotes the modified Bessel function of the first kind
    and order j.
    
    As :math:`\lim \kappa\rightarrow\infty`, the values of :math:`I_0(\kappa)`,
    :math:`I_1(\kappa)`, and in fact, :math:`I_\alpha(\kappa)` for any order
    :math:`\alpha`, approach the same value. Therefore, for large concentrations,

    .. math:
        \lim_{\kappa \rightarrow \infty} I_1(\kappa)/I_0(\kappa) = 1

    Parameters:
        mean_direction: ndarray, shape (..., 2)
        concentration: ndarray, shape (...,)
    
    Returns:
        mean: ndarray, shape (..., 2)
    """

    resultant_length = tfp.math.bessel_iv_ratio(1, concentration)
    resultant_length = jnp.nan_to_num(resultant_length, nan=1.0,
                                      posinf=jnp.inf, neginf=-jnp.inf)

    return resultant_length[..., None] * mean_direction

def vmf_mean_3d(mean_direction, concentration):
    """Calculate the mean of a 3D vMF distribution.
    
    The mean is given by
    .. math:
         \mathbb{E}[p(u)] = (\coth(\kappa) - \frac{1}{\kappa}) \nu

    Source:
        Hillen, T., Painter, K., Swan, A., and Murtha, A. 
        "Moments of von Mises and Fisher distributions and applications."
        Mathematical Biosciences and Engineering, 2017, 14(3):673-694.
        doi: 10.3934/mbe.2017038

    Parameters:
        mean_direction: ndarray, shape (..., 3)
        concentration: ndarray, shape (...,)
    
    Returns:
        mean: ndarray, shape (..., 3)
    """

    resultant_length = coth(concentration) - 1./concentration
    return resultant_length[...,None] * mean_direction

def log_vmf_normalizer_2d(concentration):
    """Calculate the log normalizer of the 2D vMF distribution:

    The normalizing constant for the 2D vMF distribution is given by
    .. math:
        C_2(\kappa) = \frac{1}{2\pi I_0(\kappa)},

    where :math:`I_0` is the modifid Bessel function of the first kind of
    order 0.

    As :math:`\kappa\rightarrow 0`, then :math:`I_0(\kappa) \rightarrow 1`
    and :math:`\log I_0 (\kappa) \rightarrow 0`. So when the vMF distribution is
    uniform over the circle, namely when :math:`\kappa=0` or is otherwise small,
    then log normalizer is the negative entropy of a uniform distribution
    over the circle:
    .. math:
        \log C_3(\kappa=0) = -log(2 \pi).

    As :math:`\kappa\rightarrow \infty`, an exponential approximation can be
    used to estimate the value of :math:`I_0(\kappa)`. The logarithm of this
    function is used to calculate the log normalizer for large concentrations.
    """

    log_i0 = jnp.log(jax.scipy.special.i0(concentration))
    log_i0 = jnp.where(jnp.isposinf(log_i0),
                       log_bessel_iv_asymptotic(concentration),
                       log_i0
                      )
    return - jnp.log(2 * jnp.pi) - log_i0

def log_vmf_normalizer_3d(concentration):
    """Calculate the log normalizer of the 3D vMF distribution:

    The normalizing constant of the 3D vMF distribution is given by 
    .. math:
        C_3(\kappa) = \frac{\kappa}{2 \pi \sinh(\kappa)}.

    As :math:`\kappa\rightarrow 0`, then :math:`\sinh(\kappa) \rightarrow \kappa`.
    So when the vMF distribution is uniform over the sphere, namely when
    :math:`\kappa=0` or is otherwise small, then log normalizer is the
    negative entropy of a uniform distribution over the sphere,
    .. math:
        \log C_3(\kappa=0) = -log(4 \pi).
    """
    log_c = - jnp.log(4 * jnp.pi) * jnp.ones_like(concentration)

    log_c += jnp.where(concentration >= CONCENTRATION_REGULARIZER,
                       jnp.log(concentration) - log_sinh(concentration),
                       jnp.zeros_like(concentration) 
                      )
    return log_c

# ---------------------------------------------------------------------------

class VMFGFunctionFactory():
    def __init__(self):
        self._functions = {
            2: {},
            3: {}
        }
    
    def register_function(self, function_name, dim, function_object):
        self._functions[dim][function_name] = function_object
    
    def get_function(self, function_name, dim):
        """Return the specified function object given distribution dimension."""
        dim_specific_functions = self._functions.get(dim)
        if dim_specific_functions is None:
            raise ValueError(dim)

        function_object = dim_specific_functions.get(function_name)
        if function_name is None:
            raise ValueError(function_name)
        return function_object

vmfg_factory = VMFGFunctionFactory()
vmfg_factory.register_function("VMF_MEAN", 2, vmf_mean_2d)
vmfg_factory.register_function("VMF_MEAN", 3, vmf_mean_3d)
vmfg_factory.register_function("LOG_VMF_NORMALIZER", 2, log_vmf_normalizer_2d)
vmfg_factory.register_function("LOG_VMF_NORMALIZER", 3, log_vmf_normalizer_3d)

# ===========================================================================

class vonMisesFisherGaussian:
    def __init__(self, mean_direction=None, concentration=None,
                 radius=None, variance=None, center=None, 
                 reinterpreted_batch_ndim=None):
        """The von Mises-Fisher-Gaussian distribution.

        Parameters:
            mean_direction: ndarray, shape (B1,... Bn, D)
              A unit vector indicating the mode of the distribution, or the unit
              direction of the mean. NOTE: `D` is currently restricted to {2,3}
            concentration: ndarray, shape (B1,... Bn)
              The level of concentration of samples around the `mean_direction`.
            radius: ndarray, shape (B1,... Bn)
              Radius of the sphere on which the data is supported.
            variance: ndarray, shape (B1,... Bn)
              Variance of data points about the surface of the sphere.
            center: ndarray, shape (B1,... Bn, D)
              Center of the sphere on which data is supported. [default: origin]
            reinterpreted_batch_ndim: integer
              Specifies the number of batch dims to be absorbed into event dim.
              [default: 0] See Notes for more detail about this parameter.
        
        Notes:
            The `reinterpreted_batch_ndim` parameter is inspired the parameter
            of the same name in tfp.distributions.Independent, which allows for
            the representation of a collection of independent, non-identical
            distributions as a single random variable.
            Concretely, when reinterpreted_batch_ndim = 0, we have the default
            batch_shape = (B1,...Bn) and event_shape = (D,).
            If reinterpreted_batch_ndim = 1, then we have a distribution with
            batch_shape = (B1,...Bn-1) and event_shape = (Bn, D,). Now, we have
            a collection Bn independent vMFG distributions of dimension D.
            Practically, this parameter affects the number of rightmost dims
            over which we sum the base distribution's log_prob.
        
        Source:
            Mukhopadhyay, M., Li D., and Dunson, D.
            "Estimating densities with non-linear support by using
            Fisher-Gaussian kernels." Journal of the Royal Statistical Society:
            Series B (Statistical Methodology), 2020, 82(5), 1249-1271.
            doi: 10.1111/rssb.12390
        """

        self._mean_direction = jnp.asarray(mean_direction)
        self._concentration = jnp.broadcast_to(concentration, self._mean_direction.shape[:-1])
        self._radius = jnp.broadcast_to(radius, self._mean_direction.shape[:-1])
        self._variance = jnp.broadcast_to(variance, self._mean_direction.shape[:-1])
        self._center = jnp.zeros_like(self._mean_direction) if center is None \
                            else jnp.broadcast_to(center, self._mean_direction.shape)

        # Specify batch and event shapes
        self._dim = self._mean_direction.shape[-1]
        if self._dim not in [2,3]:
            raise ValueError('Dimension not supported. Expected `mean_direction.shape[-1]` to be 2 or 3, received {}'.format(self._dim))
        
        self._reinterpreted_batch_ndim = 0 if reinterpreted_batch_ndim is None \
                                            else reinterpreted_batch_ndim
        self._event_ndim = 1 + self._reinterpreted_batch_ndim
        self._batch_shape = self._mean_direction.shape[:-self._event_ndim]
        self._event_shape = self._mean_direction.shape[-self._event_ndim:]
    
    @property
    def mean_direction(self):
        return self._mean_direction

    @property
    def concentration(self):
        return self._concentration

    @property
    def radius(self):
        return self._radius
        
    @property
    def variance(self):
        return self._variance

    @property
    def center(self):
        return self._center

    @property
    def dtype(self):
        return self.mean_direction.dtype

    @property
    def batch_shape(self):
        return self._batch_shape
    
    @property
    def event_shape(self):
        return self._event_shape

    @property
    def event_ndim(self):
        return self._event_ndim

    @property
    def dim(self):
        return self.event_shape[-1]


    # ==========================================================================

    def sample(self, sample_shape, seed):
        """Sample from vMFG distribution.
        
        The generative sampling model is given by
        ```
                u ~ vMF(mean_direction, concentration)
            x | u ~ MVN(radius * u + center, variance * I)
        ```

        Parameters:
            sample_shape: tuple
            seed : jax.random.PRNGKey
        
        Returns:
            pos_samples: ndarray, shape (*sample_shape, *batch_shape, *event_shape)
        """
        
        seed_1, seed_2 = jr.split(seed)

        # Each direction sample is drawn independently from parameterized vMF distribution
        vmf_samples = \
            tfp.distributions.VonMisesFisher(self.mean_direction, self.concentration).sample(sample_shape, seed_1)
        
        # Each position sample is is located at `radius * u + center`,
        # with diagonal covariance specified by `variance`
        pos_samples = jnp.sqrt(self.variance)[...,None] * jr.normal(seed_2, shape=vmf_samples.shape, dtype=self.dtype)
        pos_samples += self.radius[...,None] * vmf_samples + self.center  # Add mean

        return pos_samples

    # ==========================================================================

    def vmf_mean(self,):
        """Calculate the mean of the vMF distribution associated with this vMFG distribution instance."""
        _vmf_mean = vmfg_factory.get_function("VMF_MEAN", self.dim)
        return _vmf_mean(self.mean_direction, self.concentration)

    def log_vmf_normalizer(self, concentration):
        """Calculate the vMF log normalization constant corresponding to this vMFG instance's dimension."""
        _log_vmf_normalizer = vmfg_factory.get_function("LOG_VMF_NORMALIZER", self.dim)
        return _log_vmf_normalizer(concentration)

    # ==========================================================================

    def log_prob(self, delta):
        """Calculate log probability of center-subtracted samples `delta`."""
        return self._log_prob(delta)

    def log_prob_given_c(self, x):
        """Calculate log probability of samples `x` about their center."""
        delta = x - self.center
        return self._log_prob(delta)

    def _log_prob(self, delta):
        """Calculate log probability of center-subtracted samples under vMFG distribution

        Parameters:
            delta: ndarray, shape (...,B1,...Bn,E1,...Em,D))
        
        Returns:
            log_p: ndarray, shape(...,B1,...Bn)

        Notes:
        The probabiity density function of the vMFG distribution is given by
        .. math:
            p(x; \nu, \kappa, c, \rho, \sigma^2)
            = \frac{C_d(\kappa)}{C_d(\Vert \kappa \nu - (x-c) \rho/\sigma^2 \Vert_2)}
            \exp{-\frac{1}{2\sigma^2}((x-c)^2 + \rho^2)}.
        """
        
        D = delta.shape[-1]

        # shape (...,B1,...Bn,E1,....Em)
        conc_tilde = self.mean_direction * self.concentration[...,None]
        conc_tilde += delta * (self.radius/self.variance)[...,None]
        conc_tilde = jnp.linalg.norm(conc_tilde, axis=-1)

        log_p = self.log_vmf_normalizer(self.concentration)
        log_p -= self.log_vmf_normalizer(conc_tilde)
        log_p -= 0.5 * D * jnp.log(2*jnp.pi*self.variance)

        log_p -= 0.5 * (jnp.linalg.norm(delta, axis=-1)**2 + self.radius**2)/self.variance

        # Now add dummy axis to generalize to reinterpreted_batch_ndim >0 cases
        log_p = log_p[...,None]
        reduce_axes = tuple(-(1+onp.arange(self.event_ndim)))
        
        # shape (..., B1,...,Bn,)
        return jnp.sum(log_p, axis=reduce_axes)
    