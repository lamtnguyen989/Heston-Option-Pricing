#include "Heston-FFT.hpp"

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