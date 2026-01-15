#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
//#include <unistd.h>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define i Complex(0.0, 1.0) // Can't use `using` to device code conflict
#define PI 3.141592653589793
#define EPSILON 5e-15
#define HUGE 1e6
#define square(x) (x*x)


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
    HestonParameters();

    HestonParameters(double v_0, double kappa, double theta, double rho, double sigma)
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

// ------------------------------------------------------------------------------------ //
/* Namespace for Compuitational routines */
namespace Routines
{
    KOKKOS_INLINE_FUNCTION double std_normal_cdf(double x)  {return 0.5 * (1 + Kokkos::erf(x * 0.70710678118654752));}
    KOKKOS_INLINE_FUNCTION double std_normal_dist(double x) {return 0.39894228040143268 * Kokkos::exp(-0.5*square(x));}
    KOKKOS_INLINE_FUNCTION double black_scholes_call(double S, double K, double r, double sigma, double tau);
    KOKKOS_INLINE_FUNCTION double vega(double S, double K, double r, double sigma, double tau);
    KOKKOS_INLINE_FUNCTION double implied_volatility(double S, double K, double r, double tau, unsigned int max_iter, double epsilon);
};

/***
    Single Black-Scholes call price
***/
KOKKOS_INLINE_FUNCTION double Routines::black_scholes_call(double S, double K, double r, double sigma, double tau) 
{
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*square(sigma))) / (sigma * Kokkos::sqrt(tau));
    double d_minus = d_plus - sigma*Kokkos::sqrt(tau);
    return S*std_normal_cdf(d_plus) - std_normal_cdf(d_minus)*(K*Kokkos::exp(-r*tau));
}

/***
    Single vega compuation (Black-Scholes sigma sensitivity)
***/
KOKKOS_INLINE_FUNCTION double Routines::vega(double S, double K, double r, double sigma, double tau) 
{
    double d_plus = (Kokkos::log(S/K) + tau*(r + 0.5*square(sigma))) / (sigma * Kokkos::sqrt(tau));
    return S*std_normal_dist(d_plus)*Kokkos::sqrt(tau);
}

/*** 
    Single implied volatility computation (basic B-S differerence minimization routines)
***/
KOKKOS_INLINE_FUNCTION double Routines::implied_volatility(double S, double K, double r, double tau, 
                                            unsigned int max_iter=20, double epsilon=EPSILON)
{
    double vol = 0.2;
    double price_diff = HUGE;

    for (unsigned int iter = 0; iter < max_iter; iter++) {
        
        // Compute the difference between current computed implied vol vs market price 
        price_diff = black_scholes_call(S, K, r, vol, tau) - S;
        if (Kokkos::abs(price_diff) < epsilon) {break;}

        // Vega computation
        double _vega = vega(S, K, r, vol, tau);
        if (_vega < EPSILON) {_vega = EPSILON;}

        // Newton update to the volatility
        vol -= price_diff / _vega;
    }

    return vol;
}


// ------------------------------------------------------------------------------------ //
/* FFT solver object */
class Heston_FFT
{
    public:
        /* Constructors */
        Heston_FFT(double S, double r, Kokkos::View<double*> K, Kokkos::View<double*> T, HestonParameters(P))
            : S(S) , r(r), strikes(K) , maturities(T) , params(P), fft_config(FFT_config())
        {}


    private:
        /* Data fields */
        double S;                           // Spot price
        double r;                           // Risk-free rate
        Kokkos::View<double*> strikes;      // Strikes 
        Kokkos::View<double*> maturities;   // Contract maturities
        HestonParameters params;            // Parameters
        FFT_config fft_config;              // FFT config


        /* FFT pricing helpers */
        KOKKOS_INLINE_FUNCTION Complex heston_characteristic(Complex u, double K, double t) const;
        KOKKOS_INLINE_FUNCTION Complex damped_call(Complex v);
        KOKKOS_INLINE_FUNCTION double heston_single_call();
};

// ------------------------------------------------------------------------------------ //
/* main() */
// ------------------------------------------------------------------------------------ //
int main (int argc, char* argv[])
{
    // Setting up OpenMP backend for the cases when run on host only
    setenv("OMP_PROC_BIND", "spread", 1);
    setenv("OMP_PLACES", "threads", 1);

    Kokkos::initialize(argc, argv);
    {

    }
    Kokkos::finalize();
}