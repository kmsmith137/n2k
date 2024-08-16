#ifndef _N2K_HPP
#define _N2K_HPP

#include "n2k/Correlator.hpp"
#include "n2k/launch_rfimask_maker.hpp"
#include "n2k/launch_s0_kernel.hpp"

// Used internally
// #include "n2k/Correlator_Kernel.hpp"

// FIXME this will move to another .hpp file soon.
namespace n2k { namespace sk_globals {
    extern double mu_min;
    extern double mu_max;
    extern double xmin;
    extern double xmax;
    extern int nx;
    extern int ny;
    extern int n_min;
    extern double *get_bias_coeffs();
}}    // namespace n2k::sk_globals

#endif // _N2K_HPP
