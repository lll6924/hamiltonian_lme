# Marginalized Hamiltonian Monte Carlo for Linear Mixed Effects Models

### Prerequisites

The experiments are run with `python==3.12` and the packages in `requirements.txt`.

### Experiments

To reproduce the experiments, run the codes under `/npr/`. For instance
```commandline
python -m npr.dillonE1 --warm_up_steps 10000 --sample_steps 100000 --rng_key 0
```
will produce the results for vanilla HMC on the dillonE1 dataset. The results will be saved under `/result/`.

### Plotting

The codes to generate the figures in the paper are under `/plot/`.