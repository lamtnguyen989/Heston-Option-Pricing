#include "headers.hpp"
#include "Params-and-Configs.hpp"

/* Monte Carlo simulation results container */
struct MCResult
{
    double price;       // Monte-Carlo result discounted back
    double ci_lower;    // Confidence Interval Lower bound
    double ci_upper;    // Confidence Interval Upper bound 
};

/* Class for Heston Model simulations */
class HestonMC
{
    public:
        /* Constructors */
        HestonMC(double S0, double r, HestonParameters P)
            : S_0(S0) , r(r) , params(P)
            {}

        /* Simulations */
        MCResult single_EU_call_Milstein(double K, double T, unsigned int steps, unsigned int paths, unsigned int seed);
        MCResult single_EU_call_Milstein(double K, double T, unsigned int steps, unsigned int paths) {return single_EU_call_Milstein(K,T,steps,paths,12345);};

    private:
        /* MC Data fields */
        double S_0;
        double r;
        HestonParameters params;
};

MCResult HestonMC::single_EU_call_Milstein(double K, double T, unsigned int steps, unsigned int paths, unsigned int seed)
{
    // Processing parameters
    double dt = T / steps;

    // Initialize states and processes
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::View<double*> S("price_process", paths);
    Kokkos::View<double*> V("variance_process", paths);
    Kokkos::parallel_for("init_processes", paths, 
        KOKKOS_CLASS_LAMBDA(unsigned int p) {
            S(p) = S_0;
            V(p) = params.v0;
        });

    // Note simulation is serial in time but parallel in paths
    for (unsigned int t = 0; t < steps; t++) {
        Kokkos::parallel_for("Milstein_step", paths, 
            KOKKOS_CLASS_LAMBDA(unsigned int p){

                // Generating correlated random noise
                Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();
                double Z_s = generator.normal(0.0, 1.0);
                double Z_v = params.rho*Z_s + Kokkos::sqrt(1-square(params.rho)) * generator.normal(0.0, 1.0);
                rand_pool.free_state(generator);

                // Computing the evolution of the variance process
                double V_curr = Kokkos::fmax(V(p), 0.0);
                double V_next = V_curr
                            + params.kappa * (params.theta - V_curr) * dt
                            + params.sigma * Kokkos::sqrt(V_curr * dt) * Z_v
                            + 0.25 * square(params.sigma) * (square(Z_v) - 1.0) * dt;
                V(p) = Kokkos::fmax(V_next, 0.0);

                // Computing the evolution of the price process (using current-time variance value, not updated one)
                S(p) *= Kokkos::exp((r - 0.5*V_curr)*dt + Kokkos::sqrt(V_curr * dt) * Z_s);
        });
        Kokkos::fence();
    }

    // Reduce mean payoff
    double mean_payoff;
    Kokkos::parallel_reduce("compute_sum_payoffs", paths, 
        KOKKOS_CLASS_LAMBDA(unsigned int p, double& local_sum) {
            local_sum += Kokkos::fmax(S(p) - K, 0.0);
        }, Kokkos::Sum<double>(mean_payoff));
    mean_payoff /= paths;

    // Reduce variance of the payoffs
    double variance;
    Kokkos::parallel_reduce("compute_variance", paths,
        KOKKOS_CLASS_LAMBDA(const unsigned int p, double& local_sum) {
            local_sum += square(Kokkos::fmax(S(p) - K, 0.0) - mean_payoff);
        }, Kokkos::Sum<double>(variance));
    variance /= static_cast<double>(paths - 1);

    // Compute the standard error from variance
    double std_error = Kokkos::sqrt(variance) / Kokkos::sqrt(paths);

    // Compute the result
    double discount_factor = Kokkos::exp(-r * T);
    double ci_width = 1.96 * std_error * discount_factor;

    MCResult result;
    result.price = discount_factor * mean_payoff;
    result.ci_upper = result.price + ci_width;
    result.ci_lower = result.price - ci_width;
    
    return result;
}
