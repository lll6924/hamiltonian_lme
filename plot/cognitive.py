import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
plt.rcParams.update({'font.size': 22})

list  = ['dillonE1', 'dutch', 'eeg', 'english', 'gg05', 'mandarin', 'mandarin2', 'pupil', 'stroop']
rngs = [1, 2, 3, 4, 5]
priors = [7, 7, 7, 7, 9, 7, 7, 5, 9]
path = 'result'
data = []
mapping = {'O':'HMC', 'M1':'Marginalize $\\mathbf{u}_1$', 'M2':'Marginalize $\\mathbf{u}_2$'}
for i, dataset in enumerate(list):
    for exp in ['O','M1','M2']:
        path = os.path.join(f'../result/{dataset}/100000',exp)
        if os.path.exists(path):
            for rng in rngs:
                file = os.path.join(path, str(rng))
                if os.path.exists(file):
                    with open(file,'r') as f:
                        line1 = f.readline().split()
                        line2 = f.readline().split()
                        r1 = [float(l) for l in line1]
                        r2 = [float(l) for l in line2]
                        time1 = np.abs(r1[0])
                        prior = r2[:priors[i]]
                        effect = r2[priors[i]:]
                    data.append({'dataset':dataset, 'time':time1, 'Method':mapping[exp], 'prior': np.mean(prior)/time1, 'effect':np.mean(effect)/time1, 'ESS/iter': np.mean(r2) / 100000, 'iter/s':100000/time1})

data = pd.DataFrame(data)
sns.catplot(kind="bar", data = data, x = 'dataset', y = 'time', hue = 'Method',height=5, aspect=4)
plt.xlabel('Running time')
plt.ylabel('Time (s)')
plt.savefig('time.pdf')
plt.clf()


data = pd.DataFrame(data)
sns.catplot(kind="bar", data = data, x = 'dataset', y = 'prior', hue = 'Method',height=5, aspect=4)
plt.xlabel('Average ESS/s for prior variables and fixed effects')
plt.ylabel('ESS/s')
plt.savefig('prior.pdf')
plt.clf()

data = pd.DataFrame(data)
sns.catplot(kind="bar", data = data, x = 'dataset', y = 'effect', hue = 'Method',height=5, aspect=4)
plt.xlabel('Average ESS/s for random effects')
plt.ylabel('ESS/s')
plt.savefig('effect.pdf')

data = pd.DataFrame(data)
sns.catplot(kind="bar", data = data, x = 'dataset', y = 'ESS/iter', hue = 'Method',height=5, aspect=4)
plt.xlabel('Average ESS/iter for latent variables')
plt.savefig('ESSiter.pdf')

data = pd.DataFrame(data)
sns.catplot(kind="bar", data = data, x = 'dataset', y = 'iter/s', hue = 'Method',height=5, aspect=4)
plt.xlabel('Average iter/s for sampling')
plt.savefig('iters.pdf')