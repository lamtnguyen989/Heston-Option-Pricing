#include "headers.hpp"
#include "Params-and-Configs.hpp"


struct MCResult
{
    double price;       // Monte-Carlo result discounted back
    double std_dev;     // Standard deviation of result
    double ci_lower;    // Confidence Interval Lower bound
    double ci_upper;    // Confidence Interval Upper bound 
}


class HestonMC
{
    public:

        /* Constructors */
        HestonMC(double S0, double r, HestonParams P, unsigned int paths)
            : S_0(S0) , r(r) , params(P), n_paths(paths)
            {}

        /* European Call simulations */
        MCResult single_EU_call(double K, double T, unsigned int paths=1e5, unsigned int seed=12345);

    private:
        double S_0;
        double r;
        HestonParams params;
        unsigned int n_paths;
}


