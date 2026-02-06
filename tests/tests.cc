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
        Kokkos::printf("\n");

        ////////
        // Computing the vectorized MC simulations
        unsigned int n_strikes = 10;
        Kokkos::View<double*> strikes("strikes", n_strikes);
        Kokkos::parallel_for("fill_strikes", n_strikes,
            KOKKOS_LAMBDA(unsigned int k){
                strikes(k) = 80 + k*5.0;
        });
        Kokkos::fence();
        Kokkos::View<MCResult*> results = simulator.vanilla_calls_Milstein(strikes, maturity, steps, paths, 12345);

        Kokkos::View<double*, Kokkos::HostSpace> h_strikes = Kokkos::create_mirror_view(strikes);        
        Kokkos::View<MCResult*, Kokkos::HostSpace> h_results = Kokkos::create_mirror_view(results);
        Kokkos::deep_copy(h_results, results);
        Kokkos::deep_copy(h_strikes, strikes);
        for (unsigned int k = 0; k < h_results.extent(0); k++) {
            Kokkos::printf("Strike: %.2f, Maturity: %.2f\n", h_strikes(k), maturity);
            Kokkos::printf("%f\n", h_results(k).price);
            Kokkos::printf("95%% Confidence Interval: [%.2f , %.2f]\n", h_results(k).ci_lower, h_results(k).ci_upper);
            Kokkos::printf("\n");
        }
    }
    Kokkos::finalize();

    return 0;
}
