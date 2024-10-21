import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from numpyro.diagnostics import print_summary, summary

plt.rcParams.update({'font.size': 22})

table = ['s_obs', 'alpha','service','s','d','dept']
rg = [[1.15, 1.19], [2.0, 4.5], [-0.13, -0.03], [-2, 3], [-1, 2], [-1.5, 1.0]]
rngs = [1, 2, 3, 4, 5]
path = 'result'
data = {t:[] for t in table}
mapping = {'O':'No marginalization', 'M1':'Marginalize $\\mathbf{u}_1$', 'M2':'Marginalize $\\mathbf{u}_2$', 'M3':'Marginalize $\\mathbf{u}_3$','A':'Marginalize $\\mathbf{u}$'}
v_mapping = {'s_obs':"$\\sigma$", 'alpha':"$\\alpha$",'service':"$\\beta$",'s':"$u_{1,1}$",'d':"$u_{2,1}$",'dept':"$u_{3,1}$"}
exps = ['O','M1','M2', 'M3','A']
for exp in exps:
    print(exp)
    sums1 = []
    sums2 = []
    sums3 = []
    sums4 = []
    for rng in rngs:
        times = []
        path = os.path.join(f'../result/inst_eval/100000',exp)
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        if os.path.exists(path):
            data_file = os.path.join(path, f'{rng}.npz')
            ld = np.load(data_file, allow_pickle=True)['sample'].item()
            for t in table:
                samples = np.array([ld[t][:1000]])
                rhat = summary(samples)['Param:0']['r_hat']
                sum1 += np.sum(rhat > 1.01)
                sum2 += np.sum(rhat > 1.02)
                sum3 += np.sum(rhat > 1.05)
                sum4 += np.sum(rhat > 1.1)
                #print(t, np.min(rhat), np.max(rhat))
        sums1.append(sum1)
        sums2.append(sum2)
        sums3.append(sum3)
        sums4.append(sum4)
    print(f'{'%.2f' % np.mean(sums1)} ({'%.2f' % np.std(sums1, )})')
    print(f'{'%.2f' % np.mean(sums2)} ({'%.2f' % np.std(sums2, )})')
    print(f'{'%.2f' % np.mean(sums3)} ({'%.2f' % np.std(sums3, )})')
    print(f'{'%.2f' % np.mean(sums4)} ({'%.2f' % np.std(sums4, )})')
