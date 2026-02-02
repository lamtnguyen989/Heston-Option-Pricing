#include "headers.hpp"
#include "Params-and-Configs.hpp"


struct MCResult
{
    double price;       // Monte-Carlo result discounted back
    double std_dev;     // Standard deviation of result
    double ci_lower;    // Confidence Interval Lower bound
    double ci_upper;    // Confidence Interval Upper bound 
};


class HestonMC
{
    public:

        /* Constructors */
        HestonMC(double S0, double r, HestonParams P, unsigned int paths, unsigned int steps)
            : S_0(S0) , r(r) , params(P), n_paths(paths), n_steps(steps)
            {}

        /* Simulations */
        MCResult single_EU_call(double K, double T, unsigned int steps, unsigned int paths, unsigned int seed);

    private:
        
        /* MC Data fields */
        double S_0;
        double r;
        HestonParams params;
        unsigned int n_steps;
        unsigned int n_paths;
};

MCResult HestonMC::single_EU_call(double K, double T, unsigned int steps, unsigned int paths=1e5, seed=12345)
{
    // Initialize the random pool
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
}
