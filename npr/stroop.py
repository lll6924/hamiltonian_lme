import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_mean, init_to_sample
from jax import random, lax
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyreadr
from sklearn.preprocessing import LabelEncoder
from numpyro.diagnostics import summary, print_summary
from time import time
import click
import os
import jax
jax.config.update("jax_enable_x64", True)

def model(n_sub, n_obs, g, treatment, obs):
    sigma_u = numpyro.sample('sigma_u', dist.LKJCholesky(2))
    tau_u = numpyro.sample('tau_u', dist.HalfNormal(1), sample_shape=(2, ))
    alpha = numpyro.sample('alpha', dist.Normal(6, 1.5))
    beta = numpyro.sample('beta', dist.Normal(0, 0.01))
    sigma_alpha = numpyro.sample('sigma_alpha', dist.Normal(0, 1))
    sigma_beta = numpyro.sample('sigma_beta', dist.Normal(0, 1))
    sigma_s = numpyro.sample('sigma_s', dist.LKJCholesky(2))
    tau_s = numpyro.sample('tau_s', dist.HalfNormal(1), sample_shape=(2, ))
    s_u = jnp.matmul(jnp.diag(tau_u), sigma_u)
    s_s = jnp.matmul(jnp.diag(tau_s), sigma_s)
    s = numpyro.sample('s', dist.MultivariateNormal(jnp.zeros((2,)), scale_tril=s_s), sample_shape=(n_sub,))
    u = numpyro.sample('u', dist.MultivariateNormal(jnp.zeros((2,)), scale_tril=s_u), sample_shape=(n_sub,))

    numpyro.sample('y', dist.LogNormal(alpha + u[g][...,0] + treatment * (beta + u[g][...,1]),
                                       jnp.exp(sigma_alpha + s[g][...,0] + treatment * (sigma_beta + s[g][...,1]))), obs=obs)

@click.command()
@click.option('--rng_key', default=0,)
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
def main(rng_key, warm_up_steps, sample_steps):
    result = pyreadr.read_r('data/rdata/df_stroop.rda')
    data = result['df_stroop']
    subj_encoder = LabelEncoder()
    data["subj"] = np.array(subj_encoder.fit_transform(data["subj"].values))
    condition_encoder = LabelEncoder()
    data["condition"] = np.array(condition_encoder.fit_transform(data["condition"].values))
    np.random.seed(rng_key)
    n_obs = len(data)
    n_sub = len(subj_encoder.classes_)
    print(n_obs, n_sub)
    g = data["subj"].values

    obs = data["RT"].astype(float).values
    treatment = data["condition"].astype(float).values * 2 - 1

    start_time = time()
    nuts_kernel = NUTS(model, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps)
    sample_key = random.PRNGKey(rng_key)
    mcmc.run(sample_key, n_sub, n_obs, g, treatment, obs)
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
        print(key, np.mean(s), np.min(s), np.mean(s)/all_time,
              np.min(s)/all_time)

    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        print(
            "Number of divergences: {}".format(jnp.sum(extra_fields["diverging"]))
        )

    PATH = f'result/stroop/{sample_steps}/O'
    os.makedirs(PATH, exist_ok=True)
    output_file = f'{PATH}/{rng_key}'
    np.savez_compressed(output_file+'.npz', sample=sample)

    table = ['alpha','beta','sigma_alpha', 'sigma_beta', 'sigma_s', 'tau_s', 's' ,'sigma_u','tau_u','u']
    with open(output_file,'w') as f:
        print(all_time, jnp.sum(extra_fields["diverging"]), file=f)
        for key in table:
            s = sum[key]['n_eff']
            s = s[~np.isnan(s)]
            print(np.mean(s), end=' ',file=f)


if __name__ == '__main__':
    main()