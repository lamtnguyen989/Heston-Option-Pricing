#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosFFT.hpp>

/* Macros */
using Complex = Kokkos::complex<double>;
using exec_space = Kokkos::DefaultExecutionSpace;
#define i Complex(0.0, 1.0) // Can't use `using` to device code conflict
#define PI 3.141592653589793
#define EPSILON 5e-15
#define HUGE 1e6
#define square(x) ((x)*(x))