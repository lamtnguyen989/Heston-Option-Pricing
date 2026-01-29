# GPU-accelerated Heston Stochastic Volatility model

The project about implementing the Heston Stochastic Volatility model via the Fourier Transform of its characteristic function outlined in this [paper](https://engineering.nyu.edu/sites/default/files/2018-08/CarrMadan2_0.pdf) with a variant of characteristic function that outlined in [here](https://arxiv.org/pdf/1511.08718).

The model is further calibrated to the implied volatility surface using gradient-less [Differential Evolution](https://en.wikipedia.org/wiki/Differential_evolution) optimization method.

These are mainly exploratory methods which are somewhat feasible and reasonably fast as the model is a relatively low (5) dimensional problem and the implementation is accelerated with GPUs using the [Kokkos](https://kokkos.org/) portability library.

## Compile and run:
1. Need to install [Spack](https://spack.io/).
2. Install Kokkos via Spack.
3. Execute `run.sh` file.

## TODO:
- [] Add proper Spack enviroment config.
- [] Add proper testing framework based on the 95% Confidence Interval of the SDE formulation (discounted back to present) for the model implementation.
- [] Find how to import/feed options data onto the model for computing and testing the implied volatility calculations with the logic that is laid-out in `Routines.hpp`.
- [] Again, either import or derived implied volatility surface data to validate the calibration algorithm.
- [] Extend more capabilities after calibration to more practical stuff (i.e. Quoting and Hedging dynamics).