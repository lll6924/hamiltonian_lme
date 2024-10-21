from data.electric import *
import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, lax, jit
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import click
import os
from numpyro.diagnostics import summary,print_summary
from time import time
import jax
jax.config.update("jax_enable_x64", True)
def model(grade_pair, pair, grade, treatment, obs):
    mua = numpyro.sample('mua', dist.Normal(0., 1.), sample_shape=(n_grade_pair, ))
    sigmay = numpyro.sample('sigmay', dist.Normal(0., 1.), sample_shape = (n_grade, ))
    a = numpyro.sample('a', dist.Normal(100. * mua[grade_pair - 1], 1.),)
    b = numpyro.sample('b', dist.Normal(0., 100.), sample_shape=(n_grade, ))
    obs = numpyro.sample('y', dist.Normal(a[pair-1] + b[grade - 1] * treatment, jnp.exp(sigmay[grade - 1])), obs=obs)

def main():
    start_time = time()
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=10000, num_samples=100000)
    rng_key = random.PRNGKey(0)
    mcmc.run(rng_key, jnp.array(grade_pair), jnp.array(pair), jnp.array(grade), jnp.array(treatment), jnp.array(y))
    sample = mcmc.get_samples()
    end_time = time()
    all_time = end_time - start_time

    to_eval = {}
    for key, val in sample.items():
        to_eval[key] = np.array([val])
    sum = summary(to_eval, prob=0.9)
    print_summary(to_eval, prob=0.9)

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