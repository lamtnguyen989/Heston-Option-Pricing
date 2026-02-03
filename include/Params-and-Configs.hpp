#include "headers.hpp"

// ------------------------------------------------------------------------------------ //
/* Heston model parameters */
struct HestonParameters
{
    double v0;       // Initial variance
    double kappa;    // Mean-reversing variance process factor
    double theta;    // Long-term variance
    double rho;      // Correlation
    double sigma;    // Vol of vol

    /* Constructors */
    KOKKOS_INLINE_FUNCTION HestonParameters() {};

    KOKKOS_INLINE_FUNCTION HestonParameters(double v_0, double kappa, double theta, double rho, double sigma)
        : v0(v_0), kappa(kappa), theta(theta), rho(rho), sigma(sigma)
        {}
};

// ------------------------------------------------------------------------------------ //
/* Fast Fourier Transform config */
struct FFT_config
{
    unsigned int N; // Grid-size
    double alpha;   // Dampening factor

    FFT_config() 
        : N(8192) , alpha(2.0)
    {}
};

// ------------------------------------------------------------------------------------ //
/* Parameter bound for calibrating */
struct ParameterBounds 
{
    /* Parameters bounds */
    double v0_min, v0_max;
    double kappa_min, kappa_max;
    double theta_min, theta_max;
    double rho_min, rho_max;
    double sigma_min, sigma_max;

    /* Generation bounds (If I want to incorporate later) */

    /* Hard-setting default bounds */
    ParameterBounds()
        : v0_min(0.01) , v0_max(0.05)
        , kappa_min(0.5) , kappa_max(10.0)
        , theta_min(0.05) , theta_max(0.8)
        , rho_min(-0.999) , rho_max(0.999)
        , sigma_min(0.05) , sigma_max(0.8)
    {}
};

// ------------------------------------------------------------------------------------ //
/* Differential Evolution configurations */
struct Diff_EV_config
{
    unsigned int population_size;   // The total population size
    double crossover_prob;          // Cross-over probability
    double weight;                  // Differential weight
    unsigned int n_gen;             // Max iteration
    double tolerance;               // Conergence threshold

    Diff_EV_config(unsigned int NP, double CR, double w, unsigned int max_gen, double tol)
        : population_size(NP) , crossover_prob(CR) , weight(w) 
        , n_gen(max_gen) , tolerance(tol)
    {}

    Diff_EV_config()    // Wikipedia suggested parameters for empty constructor
        : population_size(50) , crossover_prob(0.9) , weight(0.8)
        , n_gen(100) , tolerance(EPSILON)
    {}
};

