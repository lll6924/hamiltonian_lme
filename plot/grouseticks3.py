import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import click
plt.rcParams.update({'font.size': 22})
@click.command()
@click.option('--file', default='',)
@click.option('--par', default='',)
@click.option('--log',is_flag = True,)
@click.option('--output', default = '',)
@click.option('--lower', default = -4.,)
@click.option('--upper', default = 4.,)
def main(file, par, log, output, lower, upper):
    files = ['result/grouseticks/10000/O/[0 0].npz',
             'result/grouseticks/10000/M1/[2441914641 1384938218].npz',
             'result/grouseticks/10000/M2/[4146024105  967050713].npz',
             'result/grouseticks/10000/M1R/[4146024105  967050713].npz',
             'result/grouseticks/10000/M2R/[4146024105  967050713].npz',
             'result/grouseticks/10000/R/[0 1].npz',
             ]
    titles = [
        'HMC',
        'M1',
        'M2',
        'M1, R2',
        'M2, R1',
        'R1, R2',
    ]
    titles2 = [
        'divergence = 54',
        'divergence = 32',
        'divergence = 0',
        'divergence = 27',
        'divergence = 0',
        'divergence = 12',
    ]
    samples = np.load(files[0], allow_pickle=True)['sample'].item()
    s = samples[par]
    if len(s.shape) == 1:
        s = np.expand_dims(s, 1)
    fig, axes = plt.subplots(1, len(files), figsize=(30, 5))
    for row, file in enumerate(files):
        samples = np.load(file, allow_pickle=True)['sample'].item()
        s = samples[par]
        if len(s.shape) == 1:
            s = np.expand_dims(s, 1)
        s = np.array(s)
        if log:
            s = np.log(s)

        data = []
        for i, x in enumerate(s[...,0][:1000],):
            data.append({'iteration':i, 'parameter':x})
        data = pd.DataFrame(data)
        ax = axes[row]
        #ax.set_xlim([-15,15])
        ax.set_ylim([lower, upper])
        #ax.set_xlim([-15, 10])
        #ax.set_ylim([-15, 15])
        sns.lineplot(data=data, x='iteration', y='parameter', ax=ax, linewidth = 0.5)
        if row>0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('$\\sigma_2$')
        ax.set_xlabel('iteration')
        ax.set_title(titles[row])
    #plt.show()
    plt.tight_layout()
    plt.savefig(output)
if __name__ == '__main__':
    main()