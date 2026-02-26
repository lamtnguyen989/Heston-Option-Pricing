#pragma once

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>
#include <string>
#include <type_traits>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define i Complex(0.0, 1.0) // Can't use `using` to device code conflict
#define PI 3.141592653589793
#define EPSILON 5e-15
#define HUGE 1e6
#define square(x) ((x)*(x))

enum OptionType {CALL, PUT};

void print_surface(Kokkos::View<double**>& surface, Kokkos::View<double*>& strikes, Kokkos::View<double*>& maturities, OptionType type)
{
    // Copy data back to host since parallelized prints are unreliable
    using surface_layout = typename std::remove_reference_t<decltype(surface)>::array_layout;
    Kokkos::View<double**, surface_layout, Kokkos::HostSpace> h_surface = Kokkos::create_mirror_view(surface);
    Kokkos::View<double*, Kokkos::HostSpace> h_strikes = Kokkos::create_mirror_view(strikes);
    Kokkos::View<double*, Kokkos::HostSpace> h_maturities = Kokkos::create_mirror_view(maturities);
    Kokkos::deep_copy(h_surface, surface);
    Kokkos::deep_copy(h_strikes, strikes);
    Kokkos::deep_copy(h_maturities, maturities);

    // Processing params
    unsigned int n_strikes = h_strikes.extent(0);
    unsigned int n_maturities = h_maturities.extent(0);
    const char* type_string;
    switch (type) {
        case CALL:
            type_string = "Call price grid";
            break;
        case PUT:
            type_string = "Put price grid";
            break;
    }

    // Print Maturity Headers
    Kokkos::printf("%s\n", type_string);
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
            Kokkos::printf(" %10.4f", h_surface(k, t));
        }
        Kokkos::printf("\n");
    }
    Kokkos::printf("\n");
}
