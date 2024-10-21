import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import click
plt.rcParams.update({'font.size': 22})
@click.command()
@click.option('--file', default='',)
@click.option('--par1', default='',)
@click.option('--par2', default='',)
@click.option('--log1',is_flag = True,)
@click.option('--log2', is_flag = True,)
@click.option('--output', default = '',)

def main(file, par1, par2, log1, log2, output):
    files = ['result/grouseticks/10000/O/[0 0].npz',
             'result/grouseticks/10000/M1/[4146024105  967050713].npz',
             'result/grouseticks/10000/M2/[4146024105  967050713].npz',
             'result/grouseticks/10000/M1R/[4146024105  967050713].npz',
             'result/grouseticks/10000/M2R/[4146024105  967050713].npz',
             'result/grouseticks/10000/R/[0 0].npz',
             ]
    titles = [
        'HMC',
        'M1',
        'M2',
        'M1, R2',
        'M2, R1',
        'R1, R2',
    ]
    samples = np.load(files[0], allow_pickle=True)['sample'].item()
    s1 = samples[par1]
    s2 = samples[par2]
    if len(s1.shape) == 1:
        s1 = np.expand_dims(s1, 1)
    if len(s2.shape) == 1:
        s2 = np.expand_dims(s2, 1)
    fig, axes = plt.subplots(1, len(files), figsize=(28, 5))
    for row, file in enumerate(files):
        samples = np.load(file, allow_pickle=True)['sample'].item()
        s1 = samples[par1]
        s2 = samples[par2]
        if len(s1.shape) == 1:
            s1 = np.expand_dims(s1, 1)
        if len(s2.shape) == 1:
            s2 = np.expand_dims(s2, 1)
        s1 = np.array(s1)
        s2 = np.array(s2)
        if log1:
            s1 = np.log(s1)
        if log2:
            s2 = np.log(s2)

        data = []
        for x1, x2 in zip(s1[...,0], s2[...,0]):
            data.append({'par1':x1, 'par2':x2})
        data = pd.DataFrame(data)
        ax = axes[row]
        ax.set_xlim([-15,15])
        ax.set_ylim([-4, 4])
        sns.kdeplot(data=data, x='par1', y='par2', ax=ax, levels=10)
        if row>0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('$\\sigma_2$')
        ax.set_xlabel('$u_{2,1}$')
        #ax.set_title(titles[row])
    #plt.show()
    plt.tight_layout()
    plt.savefig(output)
if __name__ == '__main__':
    main()