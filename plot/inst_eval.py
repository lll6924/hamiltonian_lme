import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
plt.rcParams.update({'font.size': 22})

for step in [0.7, 0.9]:
    table = ['s_obs', 'alpha','service','s','d','dept']
    rngs = [1, 2, 3, 4, 5]
    path = 'result'
    data = []
    mapping = {'O':'No marginalization', 'M1':'Marginalize $\\mathbf{u}_1$', 'M2':'Marginalize $\\mathbf{u}_2$', 'M3':'Marginalize $\\mathbf{u}_3$','A':'Marginalize $\\mathbf{u}$'}
    v_mapping = {'s_obs':"$\\sigma$", 'alpha':"$\\alpha$",'service':"$\\beta$",'s':"$\\mathbf{u}_1$",'d':"$\\mathbf{u}_2$",'dept':"$\\mathbf{u}_3$"}
    for exp in ['O','M1','M2', 'M3','A']:
        times = []
        path = os.path.join(f'result/inst_eval/100000_{step}',exp)
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
                        times.append(time1)
                    for ess, variable in zip(r2, table):
                        data.append({'Variable':v_mapping[variable], 'time':time1, 'Method':mapping[exp], 'ESS/s':ess/time1, 'ESS': ess})
        print(np.mean(times), np.std(times))

   # data = pd.DataFrame(data)
   # sns.catplot(kind="bar", data = data, x = 'Variable', y = 'ESS/s', hue = 'Method',height=5, aspect=4)
   # plt.xlabel('ESS/s with different sample strategies')
   # plt.savefig('inst_eval.pdf')
   # plt.clf()

    data = pd.DataFrame(data)
    sns.catplot(kind="bar", data = data, x = 'Variable', y = 'ESS', hue = 'Method',height=5, aspect=4)
    plt.xlabel(f'ESS with different sample strategies. Target probability is {step}')
    plt.savefig(f'inst_eval_ESS_{step}.pdf')
    plt.clf()
