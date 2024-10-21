import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import click
plt.rcParams.update({'font.size': 22})
@click.command()
@click.option('--file', default='',)
@click.option('--par1', default='',)
@click.option('--id1', default=0,)

@click.option('--par2', default='',)
@click.option('--log1',is_flag = True,)
@click.option('--log2', is_flag = True,)
@click.option('--output', default = '',)

def main(file, par1, id1, par2, log1, log2, output):
    files = ['result/grouseticks/10000_0.8/O/[0 0].npz',
             'result/grouseticks/10000/M1/[2441914641 1384938218].npz',
             'result/grouseticks/10000_0.8/M2/[4146024105  967050713].npz',
             'result/grouseticks/10000/M1R/[4146024105  967050713].npz',
             'result/grouseticks/10000_0.8/M2R/[4146024105  967050713].npz',
             'result/grouseticks/10000_0.8/R/[0 1].npz',
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
    s1 = samples[par1]
    s2 = samples[par2]
    if len(s1.shape) == 1:
        s1 = np.expand_dims(s1, 1)
    if len(s2.shape) == 1:
        s2 = np.expand_dims(s2, 1)
    fig, axes = plt.subplots(1, len(files), figsize=(30, 5))
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
        for x1, x2 in zip(s1[...,id1], s2[...,0]):
            data.append({'par1':x1, 'par2':x2})
        data = pd.DataFrame(data)
        ax = axes[row]
        ax.set_xlim([-10,40])
        ax.set_ylim([-4, 4])
        #ax.set_xlim([-15, 10])
        #ax.set_ylim([1.5, 3])
        sns.kdeplot(data=data, x='par1', y='par2', ax=ax, levels=10)
        with open(file[:-4], 'r') as f:
                l1 = f.readline()
                l2 = f.readline()
                l3 = f.readline().split()
        s1s = []
        for i in l3:
            i = int(i)
            ax.plot(s1[i,id1], s2[i,0], 'ro')
            s1s.append(s1[i])
        s1s = np.array(s1s)
        #print(np.mean(s1s, axis=0), np.std(s1s,axis=0))
        #print(np.abs((np.mean(s1s, axis=0) - np.mean(s1, axis=0))/(np.std(s1, axis=0)+np.std(s1s,axis=0))), np.std(s1s,axis=0)/np.std(s1, axis=0))
        if row>0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('$\\sigma_2$')
        ax.set_xlabel('$u_{2,'+str(id1+1)+'}$')
        ax.set_title(titles2[row])
    #plt.show()
    plt.tight_layout()
    plt.savefig(output)
if __name__ == '__main__':
    main()