#include "headers.hpp"

namespace Routines
{
    KOKKOS_INLINE_FUNCTION double std_normal_cdf(double x)  {return 0.5 * (1 + Kokkos::erf(x * 0.70710678118654752));}
    KOKKOS_INLINE_FUNCTION double std_normal_dist(double x) {return 0.39894228040143268 * Kokkos::exp(-0.5*square(x));}
    KOKKOS_INLINE_FUNCTION double black_scholes_call(double S, double K, double r, double sigma, double tau);
    KOKKOS_INLINE_FUNCTION double vega(double S, double K, double r, double sigma, double tau);
    KOKKOS_INLINE_FUNCTION double implied_volatility(double call_price, double S, double K, double r, double tau, unsigned int max_iter, double epsilon);
    Kokkos::View<double**> implied_volatility_surface(Kokkos::View<double**> call_prices, double S, Kokkos::View<double*> K, double r, Kokkos::View<double*> T);
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
KOKKOS_INLINE_FUNCTION double Routines::implied_volatility(double call_price, double S, double K, double r, double tau, 
                                            unsigned int max_iter=20, double epsilon=EPSILON)
{
    double vol = 0.2;
    double price_diff = HUGE;

    for (unsigned int iter = 0; iter < max_iter; iter++) {
        
        // Compute the difference between current computed implied vol vs market price 
        price_diff = black_scholes_call(S, K, r, vol, tau) - call_price;
        if (Kokkos::abs(price_diff) < epsilon) {break;}

        // Vega computation
        double v = vega(S, K, r, vol, tau);
        if (v < EPSILON) {v = EPSILON;}

        // Newton update to the volatility
        vol -= price_diff / v;
    }

    return vol;
}

/*** 
    Implied volatility surface routines
***/
Kokkos::View<double**> Routines::implied_volatility_surface(Kokkos::View<double**> call_prices, double S, Kokkos::View<double*> K, double r, Kokkos::View<double*> T)
{
    // Initialize variables and surface
    unsigned int n_strikes = K.extent(0);
    unsigned int n_maturities = T.extent(0);
    Kokkos::View<double**> iv_surface("iv_surface", n_strikes, n_maturities);
    
    // Compute the implied_volatility at every point on the surface
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0,0}, {n_strikes, n_maturities});
    Kokkos::parallel_for("iv_surface_computation", policy,
        KOKKOS_LAMBDA(unsigned int k, unsigned int t) {
            iv_surface(k,t) = Routines::implied_volatility(call_prices(k,t), S, K(k), r, T(t));
        });

    return iv_surface;
}
