#include "HestonMC.hpp"

int main(int argc, char** argv)
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

        double spot = 100.0;
        double risk_free_rate = 0.03;
        HestonMC simulator(spot, risk_free_rate, param);

        double strike = 85.0;
        double maturity = 1.0;
        unsigned int steps = 252;
        unsigned int paths = 1e4;
        unsigned int seed = 12345;
        MCResult result = simulator.single_EU_call_Milstein(strike, maturity, steps, paths, seed);


        Kokkos::printf("%f\n", result.discounted_price);
        Kokkos::printf("CI upper bound: %f\n", result.ci_upper);
        Kokkos::printf("CI lower bound: %f\n", result.ci_lower);
    }
    Kokkos::finalize();

    return 0;
}