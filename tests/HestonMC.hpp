#include "headers.hpp"
#include "Params-and-Configs.hpp"

/* Monte Carlo simulation results container */
struct MCResult
{
    double discounted_price;    // Monte-Carlo result discounted back
    double std_dev;             // Standard deviation of result
    double ci_lower;            // Confidence Interval Lower bound
    double ci_upper;            // Confidence Interval Upper bound 
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
        KOKKOS_CLASS_LAMBDA(unsigned int p){
            S(p) = S_0;
            V(p) = params.v0;
        });

    // Pre-compute random noises (to avoid potential race-condition later, tho look into more)
    Kokkos::View<double**> Z_s("stock_noise", paths, steps);
    Kokkos::View<double**> Z_v("variance_uncorrelated_noise", paths, steps);
    
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> noise_gen_policy({0,0}, {paths, steps});
    Kokkos::parallel_for("Pre_compute_uncorrelated_noise", noise_gen_policy, 
        KOKKOS_CLASS_LAMBDA(unsigned int p, unsigned int t){
            Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();

            Z_s(p,t) = generator.normal(0.0,1.0);
            Z_v(p,t) = generator.normal(0.0,1.0);

            rand_pool.free_state(generator);
        });
    Kokkos::fence();
    Kokkos::parallel_for("correlating_noise", noise_gen_policy, 
        KOKKOS_CLASS_LAMBDA(unsigned int p, unsigned int t) {
            double Z_2 = Z_v(p,t);
            Z_v(p,t) = params.rho*Z_s(p,t) + Kokkos::sqrt(1-square(params.rho))*Z_2;
        });
    Kokkos::fence();

    // Note simulation is serial in time but parallel in paths (minus the noise correlation, but that is precomputed above)
    for (unsigned int t = 0; t < steps; t++) {
        Kokkos::parallel_for("Milstein_step", paths, 
            KOKKOS_CLASS_LAMBDA(unsigned int p){

                // Computing the evolution of the variance process
                double V_curr = Kokkos::fmax(V(p), 0.0);
                double V_next = V_curr
                            + params.kappa * (params.theta - V_curr) * dt
                            + params.sigma * Kokkos::sqrt(V_curr * dt) * Z_v(p,t)
                            + 0.25 * square(params.sigma) * (square(Z_v(p,t)) - 1.0) * dt;
                V(p) = Kokkos::fmax(V_next, 0.0);

                // Computing the evolution of the price process (using current-time variance value, not updated one)
                double S_curr = S(p);
                S(p) = S_curr * Kokkos::exp((r - 0.5*V_curr)*dt + Kokkos::sqrt(V_curr * dt) * Z_s(p,t));
        });
        Kokkos::fence();
    }

    // Reduce mean payoff
    double mean_payoff = 0.0;
    Kokkos::parallel_reduce("compute_sum_payoffs", paths, 
        KOKKOS_CLASS_LAMBDA(unsigned int p, double& local_sum) {
            local_sum += Kokkos::fmax(S(p) - K, 0.0);
        }, mean_payoff);
    Kokkos::fence();
    mean_payoff /= paths;

    // Reduce variance and standard deviation of payoffs
    double variance = 0.0;
    Kokkos::parallel_reduce("compute_variance", paths,
        KOKKOS_CLASS_LAMBDA(const unsigned int p, double& local_sum) {
            double payoff_diff = Kokkos::fmax(S(p) - K, 0.0) - mean_payoff;
            local_sum += square(payoff_diff);
        }, variance);
    Kokkos::fence();
    variance /= static_cast<double>(paths - 1);
    double std_error = Kokkos::sqrt(variance) / Kokkos::sqrt(static_cast<double>(paths));

    // Compute the result
    double discount_factor = Kokkos::exp(-r * T);
    double ci_width = 1.96 * std_error * discount_factor;

    MCResult result;
    result.discounted_price = discount_factor * mean_payoff;
    result.std_dev = std_error * discount_factor;
    result.ci_upper = result.discounted_price + ci_width;
    result.ci_lower = result.discounted_price - ci_width;
    
    return result;
}
