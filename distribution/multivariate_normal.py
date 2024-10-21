import numpy as np
import numpyro
import jax.numpy as jnp
import jax.scipy as jsc
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, lax
from numpyro.distributions import constraints, Distribution
from jax.scipy.stats.norm import logcdf

from numpyro.distributions.util import (
    promote_shapes,
    validate_sample,
)

class MarginalizedMultivariateNormalProportional(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covx": constraints.positive_definite,
        "covy": constraints.positive_definite,
        "c": constraints.real_matrix,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covx",
        "covy",
        "c",
    ]

    def __init__(
        self,
        loc=0.0,
        sigmax=1.0,
        sigmay=1.0,
        x_dims = None,
        y_dim = None,
        eival = None,
        eivec = None,
        g=None,
        m_x = None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))

        self.x_dims = x_dims
        self.x_dim = jnp.sum(jnp.array(x_dims))
        self.y_dim = y_dim
        sigmax = sigmax * sigmax
        sigmay = sigmay * sigmay
        self.sigmax = sigmax
        self.sigmay = sigmay
        self.eivec = eivec
        self.eival = eival
        self.g = g
        self.m_x = m_x
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]

        batch_shape = jnp.shape(loc)[:-2]
        event_shape = jnp.shape(loc)[-1:]
        self.loc = loc[..., 0]
        super(MarginalizedMultivariateNormalProportional, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        pass

    def sample_x(self, y, key):
        r = y - self.loc
        eival_sq = jnp.square(self.eival)
        mid = 1/self.sigmax + self.eival / self.sigmay
        a1 = jnp.concatenate([jnp.bincount(g, r / self.sigmay, length = x_dim) for g, x_dim in zip(self.g,self.x_dims)])
        b1 = jnp.matmul(self.eivec.transpose(), a1) /mid * self.eival
        a2 = jnp.matmul(self.eivec, b1) / self.sigmay
        mu = self.m_x + self.sigmax * (a1 - a2)
        scale = jnp.sqrt(self.sigmax - self.sigmax * self.sigmax * self.eival / self.sigmay + eival_sq / mid * self.sigmax * self.sigmax / self.sigmay / self.sigmay)
        tril = self.eivec * scale
        sample = dist.MultivariateNormal(mu, scale_tril=tril).sample(key)
        return sample
    @validate_sample
    def log_prob(self, value):
        u = value - self.loc
        a1 = jnp.sum(u / self.sigmay * u)
        mid = 1/self.sigmax + self.eival / self.sigmay
        counts = [jnp.bincount(g, u / self.sigmay, length=x_dim) for g, x_dim in zip(self.g,self.x_dims)]
        x = jnp.concatenate(counts)
        b = jnp.matmul(self.eivec.transpose(), x)
        a2 = jnp.sum(b / mid * b)
        M =  a1 - a2
        logdet = jnp.sum(jnp.log(mid))
        logdet += jnp.log(self.sigmax) * self.x_dim + jnp.log(self.sigmay) * self.y_dim
        half_log_det = 0.5 * logdet
        normalize_term = half_log_det + 0.5 * self.y_dim * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * M - normalize_term

class MarginalizedMultivariateNormal(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covx": constraints.positive_definite,
        "covy": constraints.positive_definite,
        "c": constraints.real_matrix,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covx",
        "covy",
        "c",
    ]

    def __init__(
        self,
        loc=0.0,
        covx=1.0,
        covy=1.0,
        c=1.0,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))

        if jnp.ndim(covx) == 0:
            (covx, ) = promote_shapes(covx, shape=(1,))
        if jnp.ndim(covy) == 0:
            (covy, ) = promote_shapes(covy, shape=(1,))
        if jnp.ndim(c) == 0:
            (c, ) = promote_shapes(c, shape=(1, 1,))
        self.covx = covx
        self.covy = covy
        self.c = c
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]

        batch_shape = lax.broadcast_shapes(
            jnp.shape(loc)[:-2], jnp.shape(self.c)[:-2]
        )
        event_shape = jnp.shape(loc)[-1:]
        self.loc = loc[..., 0]
        self.x_dim = covx.shape[-1]
        self.y_dim = covy.shape[-1]
        super(MarginalizedMultivariateNormal, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        pass

    def woodbury(self, u, v):
        a1 = jnp.sum(u / self.covy * v)
        i = jnp.linalg.inv(jnp.diag(1/self.covx)+jnp.matmul(self.c.transpose() / self.covy, self.c ))
        b1 = jnp.matmul(u/self.covy,self.c)
        b2 = jnp.matmul(self.c.transpose(), v / self.covy)
        a2 = jnp.matmul(jnp.matmul(b1, i), b2)
        return a1 - a2

    def logdet(self):
        _, logdet = jnp.linalg.slogdet(jnp.diag(1/self.covx)+jnp.matmul(self.c.transpose() / self.covy, self.c ))
        return jnp.sum(jnp.log(self.covx)) + jnp.sum(jnp.log(self.covy)) + logdet

    @validate_sample
    def log_prob(self, value):
        M = self.woodbury(value - self.loc, value - self.loc)
        half_log_det = 0.5 * self.logdet()
        normalize_term = half_log_det + 0.5 * self.covy.shape[-1] * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * M - normalize_term


class MarginalizedMultivariateNormalGroup(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covx": constraints.positive_definite,
        "covy": constraints.positive_definite,
        "c": constraints.real_matrix,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covx",
        "covy",
        "c",
    ]

    def __init__(
        self,
        loc=0.0,
        sigmax=1.0,
        sigmay=1.0,
        g=None,
        x_dim = None,
        y_dim = None,
        m_x = None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        self.y_dim = y_dim
        self.x_dim = x_dim
        if jnp.ndim(sigmax) == 0:
            sigmax = jnp.full(self.x_dim, sigmax)
        if jnp.ndim(sigmay) == 0:
            sigmay = jnp.full(self.y_dim, sigmay)
        if jnp.ndim(g) == 0:
            (g, ) = promote_shapes(g, shape=(1,))
        sigmax = sigmax * sigmax
        sigmay = sigmay * sigmay
        self.covx = sigmax
        self.covy = sigmay
        self.g = g
        self.m_x = m_x
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]

        batch_shape = jnp.shape(loc)[:-2]
        event_shape = jnp.shape(loc)[-1:]
        self.loc = loc[..., 0]
        super(MarginalizedMultivariateNormalGroup, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        pass

    def sample_x(self, y, key):
        r = y - self.loc
        a1 = r / self.covy
        tm = jnp.bincount(self.g, 1 / self.covy, length=self.x_dim)
        mid = 1 / self.covx + tm
        a2 = (jnp.bincount(self.g, a1, length = self.x_dim) / mid)[self.g] / self.covy
        mu = self.m_x + self.covx * jnp.bincount(self.g, a1 - a2, length = self.x_dim)

        covs = self.covx - self.covx * tm * self.covx + self.covx * tm / mid * tm * self.covx
        return dist.Normal(mu, jnp.sqrt(covs)).sample(key)

    def woodbury(self, u, v):
        a1 = jnp.sum(u / self.covy * v)
        mid = 1/self.covx + jnp.bincount(self.g, 1/self.covy, length=self.x_dim)
        b = jnp.bincount(self.g, u/self.covy, length = self.x_dim)
        a2 = jnp.sum(b / mid * b)
        return a1 - a2

    def logdet(self):
        logdet = jnp.sum(jnp.log(1/self.covx + jnp.bincount(self.g, 1/self.covy, length=self.x_dim)))
        return jnp.sum(jnp.log(self.covx)) + jnp.sum(jnp.log(self.covy)) + logdet

    @validate_sample
    def log_prob(self, value):

        u = value - self.loc
        a1 = jnp.sum(u / self.covy * u)
        mid = 1/self.covx + jnp.bincount(self.g, 1/self.covy, length=self.x_dim)
        b = jnp.bincount(self.g, u/self.covy, length = self.x_dim)
        a2 = jnp.sum(b / mid * b)
        M = a1 - a2

        logdet = jnp.sum(jnp.log(self.covx)) + jnp.sum(jnp.log(self.covy)) + jnp.sum(jnp.log(mid))

        half_log_det = 0.5 * logdet
        normalize_term = half_log_det + 0.5 * self.covy.shape[-1] * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * M - normalize_term

class MarginalizedMultivariateNormalGroupCoeff(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covx": constraints.positive_definite,
        "covy": constraints.positive_definite,
        "c": constraints.real_matrix,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covx",
        "covy",
        "c",
    ]

    def __init__(
        self,
        loc=0.0,
        sigmax=1.0,
        sigmay=1.0,
        g=None,
        treatment = None,
        x_dim = None,
        y_dim = None,
        m_x = None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        self.y_dim = y_dim
        self.x_dim = x_dim
        if jnp.ndim(sigmax) == 0:
            sigmax = jnp.full(self.x_dim, sigmax)
        if jnp.ndim(sigmay) == 0:
            sigmay = jnp.full(self.y_dim, sigmay)
        if jnp.ndim(g) == 0:
            (g, ) = promote_shapes(g, shape=(1,))
        sigmay = sigmay * sigmay
        self.covx = jnp.matmul(sigmax, sigmax.transpose())
        self.covx_inv = jnp.linalg.inv(self.covx)
        self.covy = sigmay
        self.treatment = treatment
        self.g = g
        self.m_x = m_x
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]

        batch_shape = jnp.shape(loc)[:-2]
        event_shape = jnp.shape(loc)[-1:]
        self.loc = loc[..., 0]
        super(MarginalizedMultivariateNormalGroupCoeff, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        pass

    def sample_x(self, y, key):
        r = y - self.loc
        a1 = r / self.covy
        z1 = jnp.bincount(self.g, 1 / self.covy, length=self.x_dim)
        z2 = jnp.bincount(self.g, self.treatment / self.covy, length=self.x_dim)
        z3 = jnp.bincount(self.g, self.treatment * self.treatment / self.covy, length=self.x_dim)
        c_half = (self.covx_inv[0][1] + self.covx_inv[1][0]) / 2
        o1 = self.covx_inv[0][0] + z1
        o2 = c_half + z2
        o3 = c_half + z2
        o4 = self.covx_inv[1][1] + z3
        m1, m2, m3, m4 = self.inv_block(o1, o2, o3, o4)
        m_half = (m2 + m3) / 2
        m2 = m_half
        m3 = m_half

        b1 = jnp.bincount(self.g, r / self.covy, length=self.x_dim)
        b2 = jnp.bincount(self.g, r / self.covy * self.treatment, length=self.x_dim)

        c1 = m1 * b1 + m2 * b2
        c2 = m3 * b1 + m4 * b2

        a2 = (c1[self.g] + c2[self.g] * self.treatment) / self.covy

        d1 = jnp.bincount(self.g, a1 - a2, length=self.x_dim)
        d2 = jnp.bincount(self.g, (a1 - a2) * self.treatment, length=self.x_dim)
        covx_half = (self.covx[0][1] + self.covx[1][0]) / 2

        mu = self.m_x + jnp.array(
            [self.covx[0][0] * d1 + covx_half * d2, covx_half * d1 + self.covx[1][1] * d2]).transpose()

        e1, e2, e3, e4 = self.mul_block(*self.mul_block(z1, z2, z2, z3, m1, m2, m3, m4), z1, z2, z2, z3)
        e_half = (e2 + e3) / 2
        #f1, f2, f3, f4 = self.mul_block(*self.mul_block(*flattened, e1 - z1, e_half - z2, e_half - z2, e4 - z3),
        #                                *flattened)

        g1 = self.covx_inv[0][0] + e1 - z1
        g2 = c_half + e_half - z2
        g3 = self.covx_inv[1][1] + e4 - z3
        g = jnp.linalg.cholesky(jnp.array([g1, g2, g2, g3]).transpose().reshape((self.x_dim, 2, 2)))
        c = jnp.matmul(self.covx, g)

        #covs1 = self.covx[0][0] + f1
        #covs2 = self.covx[0][1] + f2
        #covs3 = self.covx[1][0] + f3
        #covs4 = self.covx[1][1] + f4
        #covs_half = (covs2 + covs3) / 2
        #covs = jnp.array([covs1, covs_half, covs_half, covs4]).transpose().reshape((self.x_dim, 2, 2))
        return dist.MultivariateNormal(mu, scale_tril=c).sample(key)

    def inv_block(self, a, b, c, d):
        e = c / a
        f = 1/(d - e * b)
        return 1 / a + b / a * f * e, -b / a * f, -f * e, f

    def mul_block(self, a1, b1, c1, d1, a2, b2, c2, d2):
        return a1 * a2 + b1 * c2, a1 * b2 + b1 * d2, c1 * a2 + d1 * c2, c1 * b2 + d1 * d2


    @validate_sample
    def log_prob(self, value):
        u = value - self.loc

        a1 = jnp.sum(u / self.covy * u)
        o1 = self.covx_inv[0][0] + jnp.bincount(self.g, 1/self.covy, length=self.x_dim)
        o2 = self.covx_inv[0][1] + jnp.bincount(self.g, self.treatment/self.covy, length=self.x_dim)
        o3 = self.covx_inv[1][0] + jnp.bincount(self.g, self.treatment/self.covy, length=self.x_dim)
        o4 = self.covx_inv[1][1] + jnp.bincount(self.g, self.treatment * self.treatment / self.covy, length = self.x_dim)
        m1, m2, m3, m4 = self.inv_block(o1, o2, o3, o4)

        b1 = jnp.bincount(self.g, u/self.covy, length = self.x_dim)
        b2 = jnp.bincount(self.g, u/self.covy * self.treatment, length = self.x_dim)
        a2 = jnp.sum(b1 * b1 * m1 + b1 * b2 * (m2 + m3) + b2 * b2 * m4)

        logdet = jnp.sum(jnp.log(o1 * o4 - o2 * o3))
        _, logdetx = jnp.linalg.slogdet(self.covx)

        half_log_det = 0.5 * (logdetx * self.x_dim + jnp.sum(jnp.log(self.covy)) + logdet)
        normalize_term = half_log_det + 0.5 * self.covy.shape[-1] * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * (a1 - a2) - normalize_term

class MarginalizedMultivariateLogNormalGroup(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        "covx": constraints.positive_definite,
        "covy": constraints.positive_definite,
        "c": constraints.real_matrix,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covx",
        "covy",
        "c",
    ]

    def __init__(
        self,
        loc=0.0,
        sigmax=1.0,
        sigmay=1.0,
        g=None,
        x_dim = None,
        y_dim = None,
        m_x = None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        self.y_dim = y_dim
        self.x_dim = x_dim
        if jnp.ndim(sigmax) == 0:
            sigmax = jnp.full(self.x_dim, sigmax)
        if jnp.ndim(sigmay) == 0:
            sigmay = jnp.full(self.y_dim, sigmay)
        if jnp.ndim(g) == 0:
            (g, ) = promote_shapes(g, shape=(1,))
        sigmax = sigmax * sigmax
        sigmay = sigmay * sigmay
        self.covx = sigmax
        self.covy = sigmay
        self.g = g
        self.m_x = m_x
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]

        batch_shape = jnp.shape(loc)[:-2]
        event_shape = jnp.shape(loc)[-1:]
        self.loc = loc[..., 0]
        super(MarginalizedMultivariateLogNormalGroup, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        pass

    def sample_x(self, y, key):
        y = jnp.log(y)
        r = y - self.loc
        a1 = r / self.covy
        tm = jnp.bincount(self.g, 1 / self.covy, length=self.x_dim)
        mid = 1 / self.covx + tm
        a2 = (jnp.bincount(self.g, a1, length = self.x_dim) / mid)[self.g] / self.covy
        mu = self.m_x + self.covx * jnp.bincount(self.g, a1 - a2, length = self.x_dim)

        covs = self.covx - self.covx * tm * self.covx + self.covx * tm / mid * tm * self.covx
        return dist.Normal(mu, jnp.sqrt(covs)).sample(key)

    def woodbury(self, u, v):
        a1 = jnp.sum(u / self.covy * v)
        mid = 1/self.covx + jnp.bincount(self.g, 1/self.covy, length=self.x_dim)
        b = jnp.bincount(self.g, u/self.covy, length = self.x_dim)
        a2 = jnp.sum(b / mid * b)
        return a1 - a2

    def logdet(self):
        logdet = jnp.sum(jnp.log(1/self.covx + jnp.bincount(self.g, 1/self.covy, length=self.x_dim)))
        return jnp.sum(jnp.log(self.covx)) + jnp.sum(jnp.log(self.covy)) + logdet

    @validate_sample
    def log_prob(self, value):
        y = jnp.log(value)
        M = self.woodbury(y - self.loc, y - self.loc)
        #logd, _ =  jnp.linalg.slogdet(self.sigmax * jnp.matmul(self.c, self.c.transpose()) + self.sigmay)
        half_log_det = 0.5 * self.logdet()
        normalize_term = half_log_det + 0.5 * self.covy.shape[-1] * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * M - normalize_term - jnp.sum(y)


class MarginalizedMultivariateLogNormalGroupCoeff(Distribution):
    arg_constraints = {
        "loc": constraints.real_vector,
        #"covx": constraints.positive_definite,
        #"covy": constraints.positive_definite,
        #"c": constraints.real_matrix,
    }
    support = constraints.real_vector
    reparametrized_params = [
        "loc",
        "covx",
        "covy",
        "c",
    ]

    def __init__(
        self,
        loc=0.0,
        sigmax=1.0,
        sigmay=1.0,
        g=None,
        treatment = None,
        x_dim = None,
        y_dim = None,
        m_x = None,
        validate_args=None,
    ):
        if jnp.ndim(loc) == 0:
            (loc,) = promote_shapes(loc, shape=(1,))
        self.y_dim = y_dim
        self.x_dim = x_dim
        if jnp.ndim(sigmax) == 0:
            sigmax = jnp.full(self.x_dim, sigmax)
        if jnp.ndim(sigmay) == 0:
            sigmay = jnp.full(self.y_dim, sigmay)
        if jnp.ndim(g) == 0:
            (g, ) = promote_shapes(g, shape=(1,))
        sigmay = sigmay * sigmay
        self.covx = jnp.matmul(sigmax, sigmax.transpose())
        self.covx_inv = jnp.linalg.inv(self.covx)
        self.covy = sigmay
        self.treatment = treatment
        self.g = g
        self.m_x = m_x
        # temporary append a new axis to loc
        loc = loc[..., jnp.newaxis]

        batch_shape = jnp.shape(loc)[:-2]
        event_shape = jnp.shape(loc)[-1:]
        self.loc = loc[..., 0]
        super(MarginalizedMultivariateLogNormalGroupCoeff, self).__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        pass

    def sample_x(self, y, key):
        y = jnp.log(y)
        r = y - self.loc
        a1 = r / self.covy
        z1 = jnp.bincount(self.g, 1 / self.covy, length=self.x_dim)
        z2 = jnp.bincount(self.g, self.treatment / self.covy, length=self.x_dim)
        z3 = jnp.bincount(self.g, self.treatment * self.treatment / self.covy, length=self.x_dim)
        c_half = (self.covx_inv[0][1] + self.covx_inv[1][0]) / 2
        o1 = self.covx_inv[0][0] + z1
        o2 = c_half + z2
        o3 = c_half + z2
        o4 = self.covx_inv[1][1] + z3
        m1, m2, m3, m4 = self.inv_block(o1, o2, o3, o4)
        m_half = (m2 + m3) / 2
        m2 = m_half
        m3 = m_half

        b1 = jnp.bincount(self.g, r / self.covy, length=self.x_dim)
        b2 = jnp.bincount(self.g, r / self.covy * self.treatment, length=self.x_dim)

        c1 = m1 * b1 + m2 * b2
        c2 = m3 * b1 + m4 * b2

        a2 = (c1[self.g] + c2[self.g] * self.treatment) / self.covy

        d1 = jnp.bincount(self.g, a1 - a2, length=self.x_dim)
        d2 = jnp.bincount(self.g, (a1 - a2) * self.treatment, length=self.x_dim)
        covx_half = (self.covx[0][1] + self.covx[1][0]) / 2

        mu = self.m_x + jnp.array(
            [self.covx[0][0] * d1 + covx_half * d2, covx_half * d1 + self.covx[1][1] * d2]).transpose()

        e1, e2, e3, e4 = self.mul_block(*self.mul_block(z1, z2, z2, z3, m1, m2, m3, m4), z1, z2, z2, z3)
        e_half = (e2 + e3) / 2
        # f1, f2, f3, f4 = self.mul_block(*self.mul_block(*flattened, e1 - z1, e_half - z2, e_half - z2, e4 - z3),
        #                                *flattened)

        g1 = self.covx_inv[0][0] + e1 - z1
        g2 = c_half + e_half - z2
        g3 = self.covx_inv[1][1] + e4 - z3
        g = jnp.linalg.cholesky(jnp.array([g1, g2, g2, g3]).transpose().reshape((self.x_dim, 2, 2)))
        c = jnp.matmul(self.covx, g)

        # covs1 = self.covx[0][0] + f1
        # covs2 = self.covx[0][1] + f2
        # covs3 = self.covx[1][0] + f3
        # covs4 = self.covx[1][1] + f4
        # covs_half = (covs2 + covs3) / 2
        # covs = jnp.array([covs1, covs_half, covs_half, covs4]).transpose().reshape((self.x_dim, 2, 2))
        return dist.MultivariateNormal(mu, scale_tril=c).sample(key)

    def inv_block(self, a, b, c, d):
        e = c / a
        f = 1/(d - e * b)
        return 1 / a + b / a * f * e, -b / a * f, -f * e, f

    def mul_block(self, a1, b1, c1, d1, a2, b2, c2, d2):
        return a1 * a2 + b1 * c2, a1 * b2 + b1 * d2, c1 * a2 + d1 * c2, c1 * b2 + d1 * d2


    @validate_sample
    def log_prob(self, value):
        y = jnp.log(value)
        u = y - self.loc

        a1 = jnp.sum(u / self.covy * u)
        o1 = self.covx_inv[0][0] + jnp.bincount(self.g, 1/self.covy, length=self.x_dim)
        o2 = self.covx_inv[0][1] + jnp.bincount(self.g, self.treatment/self.covy, length=self.x_dim)
        o3 = self.covx_inv[1][0] + jnp.bincount(self.g, self.treatment/self.covy, length=self.x_dim)
        o4 = self.covx_inv[1][1] + jnp.bincount(self.g, self.treatment * self.treatment / self.covy, length = self.x_dim)
        m1, m2, m3, m4 = self.inv_block(o1, o2, o3, o4)

        b1 = jnp.bincount(self.g, u/self.covy, length = self.x_dim)
        b2 = jnp.bincount(self.g, u/self.covy * self.treatment, length = self.x_dim)
        a2 = jnp.sum(b1 * b1 * m1 + b1 * b2 * (m2 + m3) + b2 * b2 * m4)

        logdet = jnp.sum(jnp.log(o1 * o4 - o2 * o3))
        _, logdetx = jnp.linalg.slogdet(self.covx)

        half_log_det = 0.5 * (logdetx * self.x_dim + jnp.sum(jnp.log(self.covy)) + logdet)
        normalize_term = half_log_det + 0.5 * self.covy.shape[-1] * jnp.log(
            2 * jnp.pi
        )
        return -0.5 * (a1 - a2) - normalize_term - jnp.sum(y)