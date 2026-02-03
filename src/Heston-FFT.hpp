#include "headers.hpp"
#include "Routines.hpp"
#include "Params-and-Configs.hpp"

/* FFT solver object */
class Heston_FFT
{
    public:
        /* Constructors */
        Heston_FFT(double S, double r, Kokkos::View<double*> K, Kokkos::View<double*> T, HestonParameters P)
            : S(S) , r(r), strikes(K) , maturities(T) , params(P), fft_config(FFT_config())
        {}

        Heston_FFT(double S, double r, Kokkos::View<double*> K, Kokkos::View<double*> T, HestonParameters P, FFT_config config)
            : S(S) , r(r), strikes(K) , maturities(T) , params(P), fft_config(config)
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
        Kokkos::View<double**> heston_put_prices(HestonParameters P, bool verbose) const;
        Kokkos::View<double**> heston_put_prices(bool verbose) {return heston_put_prices(params, verbose);};

        /* Calibrating parameters */
        HestonParameters diff_EV_iv_surf_calibration(Kokkos::View<double**> iv_surface, Kokkos::View<double*> strikes, Kokkos::View<double*> maturities,
                                            ParameterBounds bounds, Diff_EV_config config, unsigned int seed);
        HestonParameters diff_EV_iv_surf_calibration(Kokkos::View<double**> iv_surface, ParameterBounds bounds, Diff_EV_config config, unsigned int seed);

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
    if (Kokkos::real(d) < 0.0) // Branch cut handling
        d = -d;
    Complex g_2 = (xi - d) / (xi + d);
    //Complex g_1 = (xi + d) / (xi - d); // g_1 = 1/g_2

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
    Kokkos::fence();

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
        Kokkos::printf("Maturity t = %.2f\n", t);
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

    // Note: Default GPU layout is LEFT or row-major in the case of matrices (2D tensor)
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
    KokkosFFT::fft(exec_space(), x, x_hat, KokkosFFT::Normalization::backward, /*axis=*/ 0);

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
        // Copy data back to host since parallelized prints are unreliable
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> h_prices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pricing_surface);
        Kokkos::View<double*, Kokkos::HostSpace> h_strikes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), strikes);
        Kokkos::View<double*, Kokkos::HostSpace> h_maturities = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), maturities);

        // Print Maturity Headers
        Kokkos::printf("Call price grid\n");
        Kokkos::printf("%18s |", "Strike / Maturity");
        for (unsigned int t = 0; t < n_maturities; t++) {
            Kokkos::printf(" %10.2f", h_maturities(t));
        }
        Kokkos::printf("\n-------------------|");
        for (unsigned int t = 0; t < n_maturities; t++) {
            Kokkos::printf("-----------");
        }
        Kokkos::printf("\n");

        // Print Strike Rows
        for (unsigned int k = 0; k < n_strikes; k++) {
            Kokkos::printf("%18.2f |", h_strikes(k)); 
            for (unsigned int t = 0; t < n_maturities; t++) {
                Kokkos::printf(" %10.4f", h_prices(k, t));
            }
            Kokkos::printf("\n");
        }
        Kokkos::printf("\n");
    }

    return pricing_surface;
}

/***
    Put prices via parity
***/
Kokkos::View<double**> Heston_FFT::heston_put_prices(HestonParameters P, bool verbose) const
{
    // Computing the call price surface 
    Kokkos::View<double**> pricing_surface = this->heston_call_prices(P, false);

    // Apply the put-call parity
    unsigned int n_strikes = strikes.extent(0);
    unsigned int n_maturities = maturities.extent(0);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> surface_policy({0,0}, {n_strikes, n_maturities});
    Kokkos::parallel_for("put_call_parity", surface_policy,
        KOKKOS_CLASS_LAMBDA(unsigned int k, unsigned int t) {
            pricing_surface(k,t) = pricing_surface(k,t) - S + strikes(k) + Kokkos::exp(-r*t);            
        });

    if (verbose) {
        // Copy data back to host since parallelized prints are unreliable
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> h_prices = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pricing_surface);
        Kokkos::View<double*, Kokkos::HostSpace> h_strikes = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), strikes);
        Kokkos::View<double*, Kokkos::HostSpace> h_maturities = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), maturities);

        // Print Maturity Headers
        Kokkos::printf("Put price grid\n");
        Kokkos::printf("%18s |", "Strike / Maturity");
        for (unsigned int t = 0; t < n_maturities; t++) {
            Kokkos::printf(" %10.2f", h_maturities(t));
        }
        Kokkos::printf("\n-------------------|");
        for (unsigned int t = 0; t < n_maturities; t++) {
            Kokkos::printf("-----------");
        }
        Kokkos::printf("\n");

        // Print Strike Rows
        for (unsigned int k = 0; k < n_strikes; k++) {
            Kokkos::printf("%18.2f |", h_strikes(k));
            for (unsigned int t = 0; t < n_maturities; t++) {
                Kokkos::printf(" %10.4f", h_prices(k, t));
            }
            Kokkos::printf("\n");
        }
        Kokkos::printf("\n");
    }
        
    return pricing_surface;
}


/***
    Calibration
***/
HestonParameters Heston_FFT::diff_EV_iv_surf_calibration(Kokkos::View<double**> iv_surface, Kokkos::View<double*> strikes, Kokkos::View<double*> maturities,
                                            ParameterBounds bounds, Diff_EV_config config, unsigned int seed)
{
    unsigned int n_strikes = strikes.extent(0);
    unsigned int n_maturities = maturities.extent(0);

    // Checking the surface dimension
    if (n_strikes != iv_surface.extent(0) || n_maturities != iv_surface.extent(1))
        Kokkos::abort("Dimensional mismatch between implied volatility surface and strikes, maturities data");

    // Variables and containers
    unsigned int population_size = config.population_size;
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::View<HestonParameters*> population("population", population_size);
    Kokkos::View<HestonParameters*> mutations("mutations", population_size);
    Kokkos::View<double*> population_losses("population_losses", population_size);
    Kokkos::View<double*> mutation_losses("mutation_losses", population_size);

    // Initialize population
    Kokkos::parallel_for("init_pop", population_size,
        KOKKOS_CLASS_LAMBDA(unsigned int k){

            // Getting the random state
            Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();

            // Initialize population randomly
            population(k).v0 = bounds.v0_min + generator.drand()*(bounds.v0_max - bounds.v0_min);
            population(k).kappa = bounds.kappa_min + generator.drand()*(bounds.kappa_max - bounds.kappa_min);
            population(k).theta = bounds.theta_min + generator.drand()*(bounds.theta_max - bounds.theta_min);
            population(k).rho = bounds.rho_min + generator.drand()*(bounds.rho_max - bounds.rho_min);
            population(k).sigma = bounds.sigma_min + generator.drand()*(bounds.sigma_max - bounds.sigma_min);

            // Free the random state
            rand_pool.free_state(generator);
        });
    Kokkos::fence();

    // Evaluate inital population performance
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> surface_policy({0,0}, {n_strikes, n_maturities});
    for (unsigned int p = 0; p < population_size; p++) {
        double pop_iv_loss = 0;
        Kokkos::View<double**> call_prices_surface_p = this->heston_call_prices(population(p), /*verbose=*/false);
        Kokkos::View<double**> population_iv_surface_p = Routines::implied_volatility_surface(call_prices_surface_p, this->S, this->strikes, this->r, this->maturities);
        Kokkos::parallel_reduce("init_pop_loss_calc", surface_policy, 
            KOKKOS_CLASS_LAMBDA(unsigned int k, unsigned int t, double& local_iv_loss) {
                local_iv_loss += square(population_iv_surface_p(k,t) - iv_surface(k,t));
            }, pop_iv_loss);
        Kokkos::fence();
        population_losses(p) = pop_iv_loss;
    }

    // Differential Evaluation loop
    for (unsigned int gen = 0; gen < config.n_gen; gen++) {

        // Mutation generation
        Kokkos::parallel_for("mutation_generation", population_size, 
            KOKKOS_CLASS_LAMBDA(unsigned int p) {

                // Initialize random state
                Kokkos::Random_XorShift64_Pool<>::generator_type generator = rand_pool.get_state();

                // Pick 3 distinct random candidate indices
                int a, b, c;
                do {a = static_cast<unsigned int>(generator.drand()*population_size);} while (a == p);
                do {b = static_cast<unsigned int>(generator.drand()*population_size);} while (b == p || a == b);
                do {c = static_cast<unsigned int>(generator.drand()*population_size);} while (c == p || c == b || c == a);

                // Random dimensionality index (within 5 Heston parameters)
                unsigned int R = static_cast<unsigned int>(generator.drand() * 5);

                // Mutate of each dimension (parameters)
                for (unsigned int j = 0; j < 5; j++) {

                    if (generator.drand(0,1) < config.crossover_prob || j == R) {
                        switch (j) {
                            case 0: mutations(p).v0 = Kokkos::clamp(population(a).v0 + config.weight*(population(b).v0 - population(c).v0), 
                                                                    bounds.v0_min, 
                                                                    bounds.v0_max); 
                                break;
                            case 1: mutations(p).kappa = Kokkos::clamp(population(a).kappa + config.weight*(population(b).kappa - population(c).kappa), 
                                                                    bounds.kappa_min, 
                                                                    bounds.kappa_max);
                                break;
                            case 2: mutations(p).theta = Kokkos::clamp(population(a).theta + config.weight*(population(b).theta - population(c).theta), 
                                                                    bounds.theta_min, 
                                                                    bounds.theta_max);
                                break;
                            case 3: mutations(p).rho = Kokkos::clamp(population(a).rho + config.weight*(population(b).rho - population(c).rho), 
                                                                    bounds.rho_min, 
                                                                    bounds.rho_max);
                                break;
                            case 4: mutations(p).sigma = Kokkos::clamp(population(a).sigma + config.weight*(population(b).sigma - population(c).sigma), 
                                                                    bounds.sigma_min, 
                                                                    bounds.sigma_max);
                                break;
                        }
                    } else {
                        switch (j) {
                            case 0: mutations(p).v0 = population(p).v0;         break;
                            case 1: mutations(p).kappa = population(p).kappa;   break;
                            case 2: mutations(p).theta = population(p).theta;   break;
                            case 3: mutations(p).rho = population(p).rho;       break;
                            case 4: mutations(p).sigma = population(p).sigma;   break;
                        }
                    }
                }

                // Release the random state
                rand_pool.free_state(generator);
            });
        Kokkos::fence();

        // Evaluate the mutations
        for (unsigned int m = 0; m < population_size; m++) {
            double mut_iv_loss = 0.0;
            Kokkos::View<double**> call_prices_surface_m = this->heston_call_prices(mutations(m), /*verbose=*/false);
            Kokkos::View<double**> population_iv_surface_m = Routines::implied_volatility_surface(call_prices_surface_m, this->S, this->strikes, this->r, this->maturities);
            Kokkos::parallel_reduce("init_pop_loss_calc", surface_policy, 
                KOKKOS_CLASS_LAMBDA(unsigned int k, unsigned int t, double& local_iv_loss) {
                    local_iv_loss += square(population_iv_surface_m(k,t) - iv_surface(k,t));
                }, mut_iv_loss);
            Kokkos::fence();

            if (mut_iv_loss < config.tolerance) {   // If the mutation is below tolerance, early breaking
                return mutations(m);
            }
            
            mutation_losses(m) = mut_iv_loss;
        }

        // Update population based on the mutations performance
        Kokkos::parallel_for("update_population", population_size,
            KOKKOS_CLASS_LAMBDA(unsigned int p) {
                if (mutation_losses(p) < population_losses(p)) {
                    population(p) = mutations(p);
                    population_losses(p) = mutation_losses(p);
                }
            });
    }

    // Pick the best parameter from the population
    Kokkos::MinLoc<double, unsigned int>::value_type bestLoc;
    Kokkos::parallel_reduce("find_best_param", population_size,
        KOKKOS_CLASS_LAMBDA(unsigned int p, Kokkos::MinLoc<double, unsigned int>::value_type& minLoc) {
            double curr_loss = population_losses(p);
            if (curr_loss < minLoc.val) {
                minLoc.val = curr_loss;
                minLoc.loc = p;
            }
        }, Kokkos::MinLoc<double, unsigned int>(bestLoc));


    // Return
    return population(bestLoc.loc);   
}

HestonParameters Heston_FFT::diff_EV_iv_surf_calibration(Kokkos::View<double**> iv_surface, 
                                                        ParameterBounds bounds=ParameterBounds(), Diff_EV_config config=Diff_EV_config(), 
                                                        unsigned int seed=12345)
{return diff_EV_iv_surf_calibration(iv_surface, this->strikes, this->maturities, bounds, config, seed);}

