#include "headers.hpp"
#include "routines.hpp"

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
/* FFT solver object */
class Heston_FFT
{
    public:
        /* Constructors */
        Heston_FFT(double S, double r, Kokkos::View<double*> K, Kokkos::View<double*> T, HestonParameters P)
            : S(S) , r(r), strikes(K) , maturities(T) , params(P), fft_config(FFT_config())
        {}

        /* Update model methods */
        void update_parameters(HestonParameters p) {params = p;}
        void update_spot(double S_new) {S = S_new;}
        void update_rate(double r_new) {r = r_new;}
        void update_strikes(Kokkos::View<double*> K) {strikes = K;}
        void update_maturities(Kokkos::View<double*> T) {maturities = T;}
        void update_FFT_config(FFT_config conf) {fft_config = conf;}

        /* Option Pricing functions */
        Kokkos::View<double**> heston_call_prices(HestonParameters p, bool verbose) const;
        Kokkos::View<double*> heston_call_prices_at_maturity(double t, bool verbose) const;
        Kokkos::View<double**> heston_call_prices(bool verbose) {return heston_call_prices(params, verbose);};

        /* Calibrating parameters */
        double iv_surface_sq_loss(HestonParameters P);
        HestonParameters diff_EV_iv_calibration(Kokkos::View<double*> strikes, Kokkos::View<double*> maturities,
                                            ParameterBounds bounds, Diff_EV_config config, unsigned int seed);


    private:
        /* Data fields */
        double S;                           // Spot price
        double r;                           // Risk-free rate
        Kokkos::View<double*> strikes;      // Strikes 
        Kokkos::View<double*> maturities;   // Contract maturities
        HestonParameters params;            // Parameters
        FFT_config fft_config;              // FFT config


        /* FFT pricing helpers */
        KOKKOS_INLINE_FUNCTION Complex heston_characteristic(Complex u, double t, HestonParameters params) const;
        KOKKOS_INLINE_FUNCTION Complex heston_characteristic(Complex u, double t) const {return heston_characteristic(u,t,params);};
        KOKKOS_INLINE_FUNCTION Complex damped_call(Complex v, double t, HestonParameters params) const;
        KOKKOS_INLINE_FUNCTION Complex damped_call(Complex v, double t) const {return damped_call(v,t,params);};
};

/***
    Heston Characteristic functions 
***/

KOKKOS_INLINE_FUNCTION Complex Heston_FFT::heston_characteristic(Complex u, double t, HestonParameters params) const
{
    // A bunch of repeated constants in the calculation
    Complex xi = params.kappa - i*params.rho*params.sigma*u;
    Complex d = Kokkos::sqrt(square(xi) + square(params.sigma)*(square(u) + i*u));
    if (Kokkos::real(d) < 0.0) { d = -d;}
    //Complex g_1 = (xi + d) / (xi - d);
    Complex g_2 = (xi - d) / (xi + d);

    // Safe guard the fraction within the exponential terms 
    Complex log_arg_frac = Kokkos::abs(1.0 - g_2) > EPSILON 
                            ? (1.0 - g_2*Kokkos::exp(-d*t)) / (1.0 - g_2) 
                            : Complex(EPSILON, EPSILON);

    Complex last_term_frac = Kokkos::abs(1.0 - g_2*Kokkos::exp(-d*t)) > EPSILON
                            ? (1.0 - Kokkos::exp(-d*t)) / (1.0 - g_2*Kokkos::exp(-d*t))
                            : Complex(EPSILON, EPSILON);

    // Exponent
    Complex exponent = i*u*(Kokkos::log(S) + r*t)
                    + (params.kappa*params.theta)/square(params.sigma) * ((xi -d)*t - 2.0*Kokkos::log(log_arg_frac))
                    + (params.v0/square(params.sigma))*(xi-d)*(last_term_frac);

    return Kokkos::exp(exponent);
}

/*** 
    Damped call price function 
***/
KOKKOS_INLINE_FUNCTION Complex Heston_FFT::damped_call(Complex v, double t, HestonParameters params) const
{
    double alpha = fft_config.alpha;
    Complex phi = heston_characteristic((v - i*(alpha + 1.0)), t, params);
    return (Kokkos::exp(-r*t) * phi) / (square(alpha) + alpha - v*v +i*(2.0*alpha + 1.0)*v);
}

/***
    Heston call prices 
***/

Kokkos::View<double*> Heston_FFT::heston_call_prices_at_maturity(double t, bool verbose=false) const
{
    // FFT setup
    unsigned int grid_points = fft_config.N;
    double alpha = fft_config.alpha;
    double eta = 0.2;   // Step-size in damped Fourier space
    double lambda = (2.0*PI) / (grid_points*eta); // Transformed step size in log-price space
    double bound = 0.5*grid_points*lambda;

    // Compute input for the shifted phased inverse Fourier transform
    Kokkos::View<Complex*> x("Fourier input", grid_points);
    Kokkos::parallel_for("FFT_input", grid_points,
        KOKKOS_CLASS_LAMBDA(const unsigned int k) {
            
            // Compute damped call price
            double v_k = eta*k;
            Complex damped = this->damped_call(Complex(v_k, 0.0), t);

            // Compute the Simpson quadrature weights
            double w_k;
            if (k == 0 || k == grid_points-1)   { w_k = 1.0/3.0;}
            else if (k % 2 == 1)                { w_k = 4.0/3.0;}
            else                                { w_k = 2.0/3.0;}

            // Modify the damped call price to get the input
            x(k) = Kokkos::exp(-i*bound*v_k) * damped * eta * w_k; 
        }
    );

    // FFT
    Kokkos::View<Complex*> x_hat("x_hat", grid_points);
    KokkosFFT::fft(exec_space(), x, x_hat);

    // Undamped and compute the option prices
    Kokkos::View<double*> prices("prices", grid_points);
    Kokkos::parallel_for("extract_prices",  strikes.extent(0),
        KOKKOS_CLASS_LAMBDA(const unsigned int k) {

            // Find the Fourier grid price index
            double log_K = Kokkos::log(strikes(k));
            int index = static_cast<int>((log_K + bound) / lambda + 0.5);

            // Undamp to get the price at this index
            if ((index < grid_points) && (index >= 0))
                prices(k) = x_hat(index).real()*Kokkos::exp(-alpha*(lambda*index - bound))/PI ;
            else
                prices(k) = 0.0;
        });

    if (verbose) {
        Kokkos::printf("Strike \t\t Call Price\n");
        Kokkos::printf("-----------------------\n");
        Kokkos::parallel_for("print_result", strikes.extent(0),
            KOKKOS_CLASS_LAMBDA(unsigned int k){
                Kokkos::printf("%.2lf \t\t %.2lf\n", strikes(k), prices(k));
        });
        Kokkos::fence();
        Kokkos::printf("\n");
    }

    return prices;
}

Kokkos::View<double**> Heston_FFT::heston_call_prices(HestonParameters p, bool verbose=false) const
{
    // Initialize variables and surface
    unsigned int n_strikes = strikes.extent(0);
    unsigned int n_maturities = maturities.extent(0);
    unsigned int n_grid_points = fft_config.N;
    double alpha = fft_config.alpha;
    Kokkos::View<double**> pricing_surface("iv_surface", n_strikes, n_maturities);
    

    // FFT setup
    double eta = 0.2;   // Step-size in damped Fourier space
    double lambda = (2.0*PI) / (fft_config.N*eta); // Transformed step size in log-price space
    double bound = 0.5*fft_config.N*lambda;

    // Note: Default GPU layout is LEFT or row-major
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> fft_policy({0,0}, {fft_config.N, n_maturities});
    Kokkos::View<Complex**> x("Fourier_input", n_grid_points, n_maturities);

    Kokkos::parallel_for("Fourier_input_calculations" , fft_policy, 
        KOKKOS_CLASS_LAMBDA(unsigned int k, unsigned int t) {
            
            // Compute damped call price
            double v_k = eta*k;
            Complex damped = this->damped_call(Complex(v_k, 0.0), maturities(t));

            // Compute the Simpson quadrature weights
            double w_k;
            if (k == 0 || k == n_grid_points-1) { w_k = 1.0/3.0;}
            else if (k % 2 == 1)                { w_k = 4.0/3.0;}
            else                                { w_k = 2.0/3.0;}

            // Modify the damped call price to get the input
            x(k, t) = Kokkos::exp(-i*bound*v_k) * damped * eta * w_k; 
        });
    
    // FFT
    Kokkos::View<Complex**> x_hat("x_hat", n_grid_points, n_maturities);
    KokkosFFT::fft(exec_space(), x, x_hat);

    // Undamp to get option call price
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> price_policy({0,0}, {n_strikes, n_maturities});
    Kokkos::parallel_for("extract_prices", price_policy,
        KOKKOS_CLASS_LAMBDA(unsigned int k, unsigned int t) {

            // Find the Fourier grid price index
            double log_K = Kokkos::log(strikes(k));
            int index = static_cast<int>((log_K + bound) / lambda + 0.5);

            // Undamp to get the price at this index
            if ((index < n_grid_points) && (index >= 0))
                pricing_surface(k, t) = x_hat(index, t).real()*Kokkos::exp(-alpha*(lambda*index - bound)) / PI;
            else
                pricing_surface(k, t) = 0.0;
        });
    
    if (verbose) {
        // TODO
    }

    return pricing_surface;
}
