#include "Heston-FFT.hpp"
#include <fstream>  // For reading data


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
        double rho = -0.7;
        double sigma = 0.3;
        HestonParameters param(v0, kappa, theta, rho, sigma);

        // Hard setting basic strikes data
        unsigned int n_strikes = 10;
        Kokkos::View<double*> strikes("strikes", n_strikes);
        Kokkos::parallel_for("fill_strikes", n_strikes,
            KOKKOS_LAMBDA(unsigned int k){
                strikes(k) = 80 + k*5.0;
        });
        Kokkos::fence();

        // Hard-seeting basic maturities data
        unsigned int n_maturities = 5;
        Kokkos::View<double*> maturities("maturities", n_maturities);

        Kokkos::parallel_for("fill_maturities", n_maturities,
            KOKKOS_LAMBDA(unsigned int k) {
                maturities(k) = static_cast<double>(k) / (n_maturities - 1) + 0.25;
            }
        );
        Kokkos::fence();


        // Create the solver based on these hard-setted data
        double spot = 100.0;
        double risk_free_rate = 0.03;
        Heston_FFT solver(spot, risk_free_rate, strikes, maturities, param);

        // Pricing surface
        bool verbose = true;
        Kokkos::View<double**> call_surface =  solver.heston_call_prices();
        Kokkos::View<double**> put_surface = solver.heston_put_prices();

        if (verbose) {
            print_surface(call_surface, strikes, maturities, CALL);
            print_surface(put_surface, strikes, maturities, PUT);
        }

        // Pricing at t=1.0
        solver.heston_call_prices_at_maturity(1.0, verbose);
        
    }
    Kokkos::finalize();
}
