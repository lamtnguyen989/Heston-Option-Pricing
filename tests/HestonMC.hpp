#include "headers.hpp"
#include "Params-and-Configs.hpp"

/* Monte Carlo simulation results container */
struct MCResult
{
    double price;       // Monte-Carlo result discounted back
    double std_dev;     // Standard deviation of result
    double ci_lower;    // Confidence Interval Lower bound
    double ci_upper;    // Confidence Interval Upper bound 
};

/* Class for Heston Model simulations */
class HestonMC
{
    public:
        /* Constructors */
        HestonMC(double S0, double r, HestonParameters P, unsigned int paths, unsigned int steps)
            : S_0(S0) , r(r) , params(P)
            {}

        /* Simulations */
        MCResult single_EU_call_Milstein(double K, double T, unsigned int steps, unsigned int paths, unsigned int seed);

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
    Kokkos::parallel_for("Pre-computed_uncorrelated_noise", noise_gen_policy, 
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
    for (unsigned int t = 1; t < steps; t++) {
        Kokkos::parallel_for("Milstein_step", steps, 
            KOKKOS_CLASS_LAMBDA(unsigned int p){
                // TODO
        });
        Kokkos::fence();
    }


    // TODO
    MCResult result;
    result.price = 0.0;
    result.std_dev = 0.0;
    result.ci_upper = 0.0;
    result.ci_lower = 0.0;
    return result;
}
