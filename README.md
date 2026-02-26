# GPU-accelerated Heston Stochastic Volatility model

The project about implementing the Heston Stochastic Volatility model via the Fourier Transform of its characteristic function outlined in this [paper](https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf) with a variant of characteristic function that outlined in [here](https://arxiv.org/pdf/1511.08718).

The model is further calibrated to the implied volatility surface using gradient-less [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) optimization method.

These are mainly exploratory methods which are somewhat feasible and reasonably fast as the model is a relatively low (5) dimensional problem and the implementation is accelerated with GPUs using the [Kokkos](https://kokkos.org/) portability library.

## Compile and run:
1. Need to install [Spack](https://spack.io/).
2. Install Kokkos via Spack.
3. Execute `run.sh` file.