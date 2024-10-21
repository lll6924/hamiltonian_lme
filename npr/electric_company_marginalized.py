from data.electric import *
import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, lax, jit
from jax.lax import scan
import pandas as pd
from numpyro.diagnostics import summary, print_summary
import click
import os
from time import time
from distribution import MarginalizedMultivariateNormalGroup
import jax
jax.config.update("jax_enable_x64", True)
def model(grade_pair, pair, grade, treatment, obs):
    mua = numpyro.sample('mua', dist.Normal(0., 1.), sample_shape=(n_grade_pair, ))
    sigmay = numpyro.sample('sigmay', dist.Normal(0., 1.), sample_shape = (n_grade, ))
    #a = numpyro.sample('a', dist.Normal(100. * mua[grade_pair - 1], 1.),)
    a = 100. * mua[grade_pair - 1]
    b = numpyro.sample('b', dist.Normal(0., 100.), sample_shape=(n_grade, ))
    obs = numpyro.sample('y', MarginalizedMultivariateNormalGroup(a[pair-1] + b[grade - 1] * treatment, 1., jnp.exp(sigmay[grade - 1]), pair - 1, n_pair, N, ), obs=obs)

@click.command()
@click.option('--rng_key', default=0,)
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
def main(rng_key, warm_up_steps, sample_steps):
    g_p = jnp.array(grade_pair)
    p = jnp.array(pair)
    g = jnp.array(grade)
    t = jnp.array(treatment)
    obs = jnp.array(y)
    start_time = time()
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps)
    sample_key, recover_key = random.split(random.PRNGKey(rng_key))
    mcmc.run(sample_key, g_p, p, g, t, obs)
    sample = mcmc.get_samples()

    mua = sample['mua']
    sigmay = sample['sigmay']
    b = sample['b']

    def recover(_, tup):
        mua, sigmay, b, key = tup
        a = 100. * mua[g_p - 1]
        return None, MarginalizedMultivariateNormalGroup(a[p-1] + b[g - 1] * t, 1., jnp.exp(sigmay[g - 1]), p - 1, n_pair, N, a).sample_x(obs, key)

    keys = random.split(recover_key, len(mua))
    _, sample['a'] = scan(jit(recover), None, (mua, sigmay, b, keys))

    end_time = time()
    all_time = end_time - start_time

    to_eval = {}
    for key, val in sample.items():
        to_eval[key] = np.array([val])
    sum = summary(to_eval, prob=0.9)
    print_summary(to_eval, prob=0.9)
    print(all_time)
    all_variables = sum.keys()
    sorted(all_variables)
    for key in all_variables:
        s = sum[key]['n_eff']
        s = s[~np.isnan(s)]
        print(key, np.mean(s), np.min(s), np.mean(s) / all_time,
              np.min(s) / all_time)
    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        print(
            "Number of divergences: {}".format(jnp.sum(extra_fields["diverging"]))
        )

if __name__ == '__main__':
    main()