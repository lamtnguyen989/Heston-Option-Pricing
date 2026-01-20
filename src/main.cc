#include "Heston-FFT.hpp"

int main (int argc, char* argv[])
{
    // Setting up OpenMP backend for the cases when run on host only
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);

    Kokkos::initialize(argc, argv);
    {
	// Hard-setting basic parameters
	double v0 = 0.04;
      	double kappa = 2.0;
       	double theta = 0.04;
        double sigma = 0.3;
       	double rho = -0.7;
	HestonParameters param(v0, kappa, theta, sigma, rho);

	// Hard setting basic strikes data
	unsigned int n_strikes = 10;
	Kokkos::View<double*> strikes("strikes", n_strikes);

	// Hard-seeting basic maturities data
	unsigned int n_maturities = 5;
	Kokkos::View<double*> maturities("maturities", n_maturities);

	// Create the solver based on these hard-setted data
	double spot = 100.0;
	double risk_free_rate = 0.03;
	Heston_FFT solver(spot, risk_free_rate, strikes, maturities, param);

	// Pricing surface
	bool verbose = false;
	solver.heston_call_prices(verbose);

    }
    Kokkos::finalize();
}
