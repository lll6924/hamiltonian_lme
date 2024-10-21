import numpy as np
import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from jax import random, lax, vmap, jit
from distribution import MarginalizedMultivariateNormalGroup
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pyreadr
from sklearn.preprocessing import LabelEncoder
from numpyro.diagnostics import summary, print_summary
from time import time
import click
import os
from numpyro.infer.reparam import TransformReparam

def model(n_brood, n_location, n_obs, g1, g2, year, height, obs):
    m_brood = numpyro.sample('m_brood', dist.Normal(0., 1.))
    s_brood = numpyro.sample('s_brood', dist.HalfCauchy(5))
    m_location = numpyro.sample('m_location', dist.Normal(0., 1.))
    s_location = numpyro.sample('s_location', dist.HalfCauchy(5))
    with numpyro.handlers.reparam(config={'brood': TransformReparam(),}):
        brood = numpyro.sample('brood',
            dist.TransformedDistribution(dist.Normal(jnp.zeros(n_brood), jnp.ones(n_brood)),
            dist.transforms.AffineTransform(m_brood, s_brood)))
    location = jnp.repeat(m_location, n_location)
    #location = numpyro.sample('location', dist.Normal(m_location,s_location), sample_shape=(n_location,))
    s_obs = numpyro.sample('s_obs', dist.HalfCauchy(5))
    y = numpyro.sample('year', dist.Normal(0, 1))
    h = numpyro.sample('height', dist.Normal(0, 1))
    numpyro.sample('y', MarginalizedMultivariateNormalGroup(brood[g1] + location[g2] + year * y + height * h, s_location, s_obs, g2, n_location, n_obs), obs=obs)

@click.command()
@click.option('--rng_key', default=0,)
@click.option('--target_prob', default=0.8,)
@click.option('--warm_up_steps', default=10000, help = 'Number of warm up samples in HMC')
@click.option('--sample_steps', default=100000, help = 'Number of samples in HMC')
def main(rng_key, target_prob, warm_up_steps, sample_steps):
    result = pyreadr.read_r('data/rdata/grouseticks.rda')
    data = result['grouseticks']
    location_encoder = LabelEncoder()
    data["LOCATION"] = np.array(location_encoder.fit_transform(data["LOCATION"].values))

    brood_encoder = LabelEncoder()
    data["BROOD"] = np.array(brood_encoder.fit_transform(data["BROOD"].values))
    np.random.seed(rng_key)
    n_obs = len(data)
    n_brood = len(brood_encoder.classes_)
    n_location = len(location_encoder.classes_)
    g1 = data["BROOD"].values
    g2 = data["LOCATION"].values
    obs = data["TICKS"].astype(float).values
    year = data["YEAR"].astype(float).values
    height = data["cHEIGHT"].astype(float).values
    time1 = time()
    nuts_kernel = NUTS(model, target_accept_prob=target_prob)
    mcmc = MCMC(nuts_kernel, num_warmup=warm_up_steps, num_samples=sample_steps)
    rng_key, recover_key = random.split(random.PRNGKey(rng_key))
    mcmc.run(rng_key, n_brood, n_location, n_obs, g1, g2, year, height, obs)
    sample = mcmc.get_samples()
    h = sample['height']
    b = sample['brood']
    m_brood = sample['m_brood']
    m_location = sample['m_location']
    s_brood = sample['s_brood']
    s_location = sample['s_location']
    s_obs = sample['s_obs']
    y = sample['year']

    def recover(h, b, m_brood, m_location, s_brood, s_location, s_obs, y, key):
        location = jnp.repeat(m_location, n_location)
        return MarginalizedMultivariateNormalGroup(b[g1] + location[g2] + year * y + height * h, s_location, s_obs, g2,
                                                   n_location, n_obs, location).sample_x(obs, key)

    keys = random.split(recover_key, len(h))
    sample['location'] = jit(vmap(recover))(h, b, m_brood, m_location, s_brood, s_location, s_obs, y, keys)
    time2 = time()
    to_eval = {}
    for key, val in sample.items():
        to_eval[key] = np.array([val])
    sum = summary(to_eval, prob=0.9)
    all_time = time2 - time1
    print(all_time)
    all_variables = sum.keys()
    sorted(all_variables)
    for key in all_variables:
        print(key, np.mean(sum[key]['n_eff']), np.min(sum[key]['n_eff']), np.mean(sum[key]['n_eff'])/all_time, np.min(sum[key]['n_eff'])/all_time)

    extra_fields = mcmc.get_extra_fields()
    if "diverging" in extra_fields:
        print(
            "Number of divergences: {}".format(jnp.sum(extra_fields["diverging"]))
        )

    PATH = f'result/grouseticks/{sample_steps}_{target_prob}/M2R/'
    os.makedirs(PATH, exist_ok=True)
    output_file = f'result/grouseticks/{sample_steps}_{target_prob}/M2R/{rng_key}'
    np.savez_compressed(output_file+'.npz', sample=sample)
    table = ['m_brood','s_brood','m_location','s_location','year','height','brood','location','s_obs']
    with open(output_file,'w') as f:
        print(all_time, jnp.sum(extra_fields["diverging"]), file=f)
        for key in table:
            print(np.mean(sum[key]['n_eff']), end=' ',file=f)

        print(file=f)
        for id in jnp.where(extra_fields["diverging"])[0]:
            print(id, end=' ', file=f)


if __name__ == '__main__':
    main()