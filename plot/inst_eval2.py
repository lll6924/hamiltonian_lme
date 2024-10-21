import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
plt.rcParams.update({'font.size': 22})

table = ['s_obs', 'alpha','service','s','d','dept']
rg = [[1.15, 1.19], [2.0, 4.5], [-0.13, -0.03], [-2, 3], [-1, 2], [-1.5, 1.0]]
rngs = [1,]
path = 'result'
data = {t:[] for t in table}
mapping = {'O':'No marginalization', 'M1':'Marginalize $\\mathbf{u}_1$', 'M2':'Marginalize $\\mathbf{u}_2$', 'M3':'Marginalize $\\mathbf{u}_3$','A':'Marginalize $\\mathbf{u}$'}
v_mapping = {'s_obs':"$\\sigma$", 'alpha':"$\\alpha$",'service':"$\\beta$",'s':"$u_{1,1}$",'d':"$u_{2,1}$",'dept':"$u_{3,1}$"}
exps = ['O','M1','M2', 'M3','A']
for exp in exps:
    print(exp)
    times = []
    path = os.path.join(f'../result/inst_eval/100000',exp)
    if os.path.exists(path):
        data_file = os.path.join(path, '1.npz')
        ld = np.load(data_file, allow_pickle=True)['sample'].item()
        for t in table:
            for i in range(10000):
                if i < 9000:
                    continue
                if len(ld[t].shape) == 2:
                    val = float(ld[t][i][0])
                else:
                    val = float(ld[t][i])
                data[t].append({'step':i, v_mapping[t]:val, 'exp':exp})

for j, t in enumerate(table):
    d = pd.DataFrame(data[t])
    fig, axes = plt.subplots(1, len(exps), figsize=(30, 5))
    for i, ax in enumerate(axes):
        sns.lineplot(data=d, x='step', y=v_mapping[t], hue='exp', hue_order=[exps[i]], ax=ax, linewidth = 0.5)
        ax.get_legend().set_visible(False)
        ax.set_ylim(rg[j])
        if i>0:
            ax.set_ylabel('')
            ax.set_yticks([])
        else:
            ax.set_ylabel(v_mapping[t])
        ax.set_title(mapping[exps[i]])
    plt.tight_layout()
    plt.savefig(f'inst_eval_trace{t}2.pdf')
    plt.clf()



