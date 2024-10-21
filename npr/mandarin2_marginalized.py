import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, HMC
from jax import random, lax, vmap, jit
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyreadr
from sklearn.preprocessing import LabelEncoder
from numpyro.diagnostics import summary, print_summary
from time import time
from distribution import MarginalizedMultivariateLogNormalGroupCoeff
import click
import os
from jax.lax import scan
import jax
jax.config.update("jax_enable_x64", True)

def model(n_sub, n_item, n_obs, g1, g2, treatment, obs):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 5))
    sigma = numpyro.sample('sigma', dist.HalfNormal(5))
    sigma_u = numpyro.sample('sigma_u', dist.LKJCholesky(2))
    tau_u = numpyro.sample('tau_u', dist.HalfNormal(5), sample_shape=(2, ))
    sigma_w = numpyro.sample('sigma_w', dist.LKJCholesky(2))
    tau_w = numpyro.sample('tau_w', dist.HalfNormal(5), sample_shape=(2, ))
    s_u = jnp.matmul(jnp.diag(tau_u), sigma_u)
    s_w = jnp.matmul(jnp.diag(tau_w), sigma_w)
    u = jnp.zeros((n_sub, 2))
#    u = numpyro.sample('u', dist.MultivariateNormal(jnp.zeros((2,)), scale_tril=s_u), sample_shape=(n_sub,))
    w = numpyro.sample('w', dist.MultivariateNormal(jnp.zeros((2,)),scale_tril=s_w), sample_shape=(n_item,))
    numpyro.sample('y', MarginalizedMultivariateLogNormalGroupCoeff(alpha + u[g1][...,0] + w[g2][...,0] + treatment * (beta + u[g1][...,1] + w[g2][...,1]), s_u, sigma, g1, treatment, n_sub, n_obs), obs=obs)

@click.command()
@click.option('--rng_key', default=0,)
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
def main(rng_key, warm_up_steps, sample_steps):
    result = pyreadr.read_r('data/rdata/df_gibsonwu2.rda')
    data = result['df_gibsonwu2']
    subj_encoder = LabelEncoder()
    data["subj"] = np.array(subj_encoder.fit_transform(data["subj"].values))
    item_encoder = LabelEncoder()
    data["item"] = np.array(item_encoder.fit_transform(data["item"].values))

    type_encoder = LabelEncoder()
    data["condition"] = np.array(type_encoder.fit_transform(data["condition"].values))

    np.random.seed(rng_key)
    n_obs = len(data)
    n_sub = len(subj_encoder.classes_)
    n_item = len(item_encoder.classes_)
    g1 = data["subj"].values
    g2 = data["item"].values
    print(n_sub, n_item,)

    obs = data["rt"].astype(float).values
    treatment = data["condition"].astype(float).values - 0.5
    print(treatment, obs)
    start_time = time()
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps)
    sample_key, recover_key = random.split(random.PRNGKey(rng_key))
    mcmc.run(sample_key, n_sub, n_item, n_obs, g1, g2, treatment, obs)
    sample = mcmc.get_samples()

    alpha = sample['alpha']
    beta = sample['beta']
    sigma = sample['sigma']
    sigma_u = sample['sigma_u']
    tau_u = sample['tau_u']
    tau_w = sample['tau_w']
    w = sample['w']
    sigma_w = sample['sigma_w']

    def recover(_, tup):
        alpha, beta, sigma, sigma_u, tau_u, sigma_w, tau_w, w, key = tup
        u = jnp.zeros((n_sub, 2))
        s_u = jnp.matmul(jnp.diag(tau_u), sigma_u)
        return None, MarginalizedMultivariateLogNormalGroupCoeff(
            alpha + u[g1][..., 0] + w[g2][..., 0] + treatment * (beta + u[g1][..., 1] + w[g2][..., 1]),
            s_u, sigma, g1, treatment, n_sub, n_obs, u).sample_x(obs, key)

    keys = random.split(recover_key, len(w))
    _, sample['u'] = scan(jit(recover), None, (alpha, beta, sigma, sigma_u, tau_u, sigma_w, tau_w, w, keys))

    end_time = time()
    all_time = end_time - start_time
    to_eval = {}
    for key, val in sample.items():
        to_eval[key] = np.array([val])
    sum = summary(to_eval, prob = 0.9)
    print_summary(to_eval, prob = 0.9)

    all_variables = sum.keys()
    sorted(all_variables)
    for key in all_variables:
        s = sum[key]['n_eff']
        s = s[~np.isnan(s)]
        print(key, np.mean(s), np.min(s), np.mean(s)/all_time, np.min(s)/all_time)
    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        print(
            "Number of divergences: {}".format(jnp.sum(extra_fields["diverging"]))
        )

    PATH = f'result/mandarin2/{sample_steps}/M1'
    os.makedirs(PATH, exist_ok=True)
    output_file = f'{PATH}/{rng_key}'
    np.savez_compressed(output_file+'.npz', sample=sample)

    table = ['alpha','beta','sigma','sigma_u','tau_u','sigma_w','tau_w','u','w']
    with open(output_file,'w') as f:
        print(all_time, jnp.sum(extra_fields["diverging"]), file=f)
        for key in table:
            s = sum[key]['n_eff']
            s = s[~np.isnan(s)]
            print(np.mean(s), end=' ',file=f)


if __name__ == '__main__':
    main()