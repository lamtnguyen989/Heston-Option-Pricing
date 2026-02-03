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
            : S_0(S0) , r(r) , params(P), n_paths(paths), n_steps(steps)
            {}

        /* Simulations */
        MCResult single_EU_call_Milstein(double K, double T, unsigned int steps, unsigned int paths, unsigned int seed);

    private:
        /* MC Data fields */
        double S_0;
        double r;
        HestonParameters params;
        unsigned int n_steps;
        unsigned int n_paths;
};

MCResult HestonMC::single_EU_call_Milstein(double K, double T, unsigned int steps, unsigned int paths=1e5, unsigned int seed=12345)
{
    // Initialize states
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    MCResult result;

    // TODO
    result.price = 0.0;
    result.std_dev = 0.0;
    result.ci_upper = 0.0;
    result.ci_lower = 0.0;
    return result;
}
