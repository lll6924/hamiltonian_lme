import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_mean
from jax import random, lax
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyreadr
from sklearn.preprocessing import LabelEncoder
import jax
jax.config.update("jax_enable_x64", True)
from time import time
from numpyro.diagnostics import print_summary, summary
import click
import os
def model(n_s, n_d, n_dept, n_obs, g0, g1, g2, studage, lectage, service, obs):
    m_s = 0#numpyro.sample('m_s', dist.Normal(0., 5.))
    s_s = 1. #numpyro.sample('s_s', dist.HalfNormal(1))
    m_d = 0#numpyro.sample('m_d', dist.Normal(0., 5.))
    s_d = 1. #numpyro.sample('s_d', dist.HalfNormal(1))
    m_dept = 0#numpyro.sample('m_dept', dist.Normal(0., 5.))
    s_dept = 1. #numpyro.sample('s_dept', dist.HalfNormal(1))
    #u =  numpyro.sample('u', dist.Normal(0,1), sample_shape=(n_s,))
    #v = numpyro.sample('v', dist.Normal(0,1), sample_shape=(n_d,))
    #s = u * s_s + m_s
    #d = v * s_d + m_d
    s = numpyro.sample('s', dist.Normal(m_s,s_s), sample_shape=(n_s,))
    d = numpyro.sample('d', dist.Normal(m_d,s_d), sample_shape=(n_d,))
    dept = numpyro.sample('dept', dist.Normal(m_dept,s_dept), sample_shape=(n_dept,))
    s_obs = numpyro.sample('s_obs', dist.HalfNormal(1))
    #stu = numpyro.sample('studage', dist.Normal(0, 1))
    #lec = numpyro.sample('lectage', dist.Normal(0, 1))
    ser = numpyro.sample('service', dist.Normal(0, 1))
    alpha = numpyro.sample('alpha', dist.Normal(0, 5))
    numpyro.sample('y', dist.Normal(s[g0] + d[g1] + alpha + dept[g2] + service * ser, s_obs), obs=obs)

@click.command()
@click.option('--rng_key', default=0,)
@click.option('--target_prob', default=0.8,)
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
def main(rng_key, target_prob, warm_up_steps, sample_steps):
    result = pyreadr.read_r('data/rdata/InstEval.rda')
    data = result['InstEval']

    s_encoder = LabelEncoder()
    data["s"] = np.array(s_encoder.fit_transform(data["s"].values))

    d_encoder = LabelEncoder()
    data["d"] = np.array(d_encoder.fit_transform(data["d"].values))

    dept_encoder = LabelEncoder()
    data["dept"] = np.array(dept_encoder.fit_transform(data["dept"].values))
    np.random.seed(rng_key)
    n_obs = len(data)
    n_s = len(s_encoder.classes_)
    n_d = len(d_encoder.classes_)
    n_dept = len(dept_encoder.classes_)
    obs = data["y"].astype(float).values
    studage = data["studage"].astype(float).values
    lectage = data["lectage"].astype(float).values
    service = data["service"].astype(float).values
    start_time = time()

    nuts_kernel = NUTS(model, max_tree_depth=12, target_accept_prob=target_prob)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps)
    sample_key = random.PRNGKey(rng_key)
    mcmc.run(sample_key, n_s, n_d, n_dept, n_obs, data["s"].values, data["d"].values, data["dept"].values, studage, lectage, service, obs)
    sample = mcmc.get_samples()
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

    PATH = f'result/inst_eval/{sample_steps}_{target_prob}/O'
    os.makedirs(PATH, exist_ok=True)
    output_file = f'{PATH}/{rng_key}'
    np.savez_compressed(output_file+'.npz', sample=sample)

    table = ['s_obs', 'alpha','service','s','d','dept']
    with open(output_file,'w') as f:
        print(all_time, jnp.sum(extra_fields["diverging"]), file=f)
        for key in table:
            s = sum[key]['n_eff']
            s = s[~np.isnan(s)]
            print(np.mean(s), end=' ',file=f)



if __name__ == '__main__':
    main()