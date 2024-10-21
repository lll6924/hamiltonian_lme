import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC, init_to_mean
from jax import random, lax, vmap, jit
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyreadr
from sklearn.preprocessing import LabelEncoder
from numpyro.diagnostics import summary, print_summary
from distribution import MarginalizedMultivariateNormalGroupCoeff
from time import time
import os
import click
from jax.lax import scan
import jax
jax.config.update("jax_enable_x64", True)

def model(n_sub, n_item, n_obs, g1, g2, treatment, obs):
    sigma_u = numpyro.sample('sigma_u', dist.LKJCholesky(2))
    tau_u = numpyro.sample('tau_u', dist.HalfNormal(1), sample_shape=(2,))
    s_u = jnp.matmul(jnp.diag(tau_u), sigma_u)
    u = jnp.zeros((n_sub, 2))
    #u = numpyro.sample('u', dist.MultivariateNormal(jnp.zeros((2,)), scale_tril=s_u), sample_shape=(n_sub,))
    sigma_v = numpyro.sample('sigma_v', dist.LKJCholesky(2))
    tau_v = numpyro.sample('tau_v', dist.HalfNormal(1), sample_shape=(2,))
    s_v = jnp.matmul(jnp.diag(tau_v), sigma_v)
    v = numpyro.sample('v', dist.MultivariateNormal(jnp.zeros((2,)), scale_tril=s_v), sample_shape=(n_item,))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 5))
    numpyro.sample('y', MarginalizedMultivariateNormalGroupCoeff(alpha + u[g1][...,0] + v[g2][...,0] + treatment * (beta + u[g1][...,1] + v[g2][...,1]), s_u, sigma, g1, treatment, n_sub, n_obs), obs=obs)
@click.command()
@click.option('--rng_key', default=0,)
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
def main(rng_key, warm_up_steps, sample_steps):
    result = pyreadr.read_r('data/rdata/df_dutch.rda')
    data = result['df_dutch']
    subj_encoder = LabelEncoder()
    data["subject"] = np.array(subj_encoder.fit_transform(data["subject"].values))
    item_encoder = LabelEncoder()
    data["item"] = np.array(item_encoder.fit_transform(data["item"].values))
    np.random.seed(rng_key)
    n_obs = len(data)
    n_sub = len(subj_encoder.classes_)
    n_item = len(item_encoder.classes_)
    g1 = data["subject"].values
    g2 = data["item"].values

    obs = data["NP1"].astype(float).values
    treatment = data["condition"].astype(float).values
    time1 = time()
    nuts_kernel = NUTS(model, init_strategy=init_to_mean)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps)
    sample_key, recover_key = random.split(random.PRNGKey(rng_key))
    mcmc.run(sample_key, n_sub, n_item, n_obs, g1, g2, treatment, obs)
    sample = mcmc.get_samples()
    alpha = sample['alpha']
    beta = sample['beta']
    sigma = sample['sigma']
    sigma_u = sample['sigma_u']
    tau_u = sample['tau_u']
    tau_v = sample['tau_v']
    v = sample['v']
    sigma_v = sample['sigma_v']

    def recover(_, tup):
        alpha, beta, sigma, sigma_u, sigma_v, tau_u, tau_v, v, key = tup
        u = jnp.zeros((n_sub, 2))
        s_u = jnp.matmul(jnp.diag(tau_u), sigma_u)
        return None, MarginalizedMultivariateNormalGroupCoeff(alpha + u[g1][...,0] + v[g2][...,0] + treatment * (beta + u[g1][...,1] + v[g2][...,1]),
                                                         s_u, sigma, g1, treatment, n_sub, n_obs, u).sample_x(obs, key)

    keys = random.split(recover_key, len(alpha))
    _, sample['u'] = scan(jit(recover), None, (alpha, beta, sigma, sigma_u, sigma_v, tau_u,tau_v, v, keys))
    to_eval = {}
    for key, val in sample.items():
        to_eval[key] = np.array([val])
    time2 = time()
    all_time = time2 - time1
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

    PATH = f'result/dutch/{sample_steps}/M1'
    os.makedirs(PATH, exist_ok=True)
    output_file = f'{PATH}/{rng_key}'
    np.savez_compressed(output_file+'.npz', sample=sample)

    table = ['alpha','beta','sigma','sigma_u','tau_u','sigma_v','tau_v','u','v']
    with open(output_file,'w') as f:
        print(all_time, jnp.sum(extra_fields["diverging"]), file=f)
        for key in table:
            s = sum[key]['n_eff']
            s = s[~np.isnan(s)]
            print(np.mean(s), end=' ',file=f)

if __name__ == '__main__':
    main()