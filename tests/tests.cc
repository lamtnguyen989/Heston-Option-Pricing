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

        // Creating simulator object
        double spot = 100.0;
        double risk_free_rate = 0.03;
        HestonMC simulator(spot, risk_free_rate, param);

        // Simulate a single vanilla call
        double strike = 85.0;
        double maturity = 1.0;
        unsigned int steps = 252;
        unsigned int paths = 3000;
        MCResult result = simulator.single_EU_call_Milstein(strike, maturity, steps, paths);

        // Output to console for now
        Kokkos::printf("Strike: %.2f, Maturity: %.2f\n", strike, maturity);
        Kokkos::printf("%f\n", result.price);
        Kokkos::printf("95%% Confidence Interval: [%.2f , %.2f]\n", result.ci_lower, result.ci_upper);
    }
    Kokkos::finalize();

    return 0;
}
