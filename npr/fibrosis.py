import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, lax
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import click
import os
import jax
from time import time
from numpyro.diagnostics import summary, print_summary
jax.config.update("jax_enable_x64", True)
def model(n_patients, n_obs, g, t, obs):
    m_a = numpyro.sample("m_a", dist.Normal(0.0, 500.0))
    s_a = numpyro.sample("s_a", dist.HalfCauchy(100.0))
    m_b = numpyro.sample("m_b", dist.Normal(0.0, 3.0))
    s_b = numpyro.sample("s_b", dist.HalfCauchy(3.0))
    s = numpyro.sample("s", dist.HalfCauchy(100.0))
    a = numpyro.sample("a", dist.Normal(m_a, s_a),  sample_shape=(n_patients, ))
    b = numpyro.sample("b", dist.Normal(m_b, s_b),  sample_shape=(n_patients, ))
    numpyro.sample('y', dist.Normal(a[g] + t * b[g], s), obs=obs)


@click.command()
@click.option('--rng_key', default=0,)
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
def main(rng_key, warm_up_steps, sample_steps):
    train = pd.read_csv('data/p_f.csv')
    np.random.seed(10)
    patient_encoder = LabelEncoder()
    train["patient_code"] = np.array(patient_encoder.fit_transform(train["Patient"].values))
    patient_code = train["patient_code"].values
    n_patients = len(np.unique(patient_code))
    n_obs = len(train["FVC"].values)

    start_time = time()
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps)
    sample_key = random.PRNGKey(rng_key)
    mcmc.run(sample_key,  n_patients, n_obs, patient_code, train["Weeks"].values, train["FVC"].values)
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