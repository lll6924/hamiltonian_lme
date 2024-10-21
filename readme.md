# Hamiltonian Monte Carlo of Marginalized Linear Mixed Effects Models

This is the codes for the NeurIPS 2024 paper titled "Hamiltonian Monte Carlo of Marginalized Linear Mixed Effects Models". More instructions will be updated later.

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