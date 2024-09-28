#ifndef _N2K_SK_GLOBALS_HPP
#define _N2K_SK_GLOBALS_HPP


namespace n2k { namespace sk_globals {
    static constexpr double mu_min = 1.0;
    static constexpr double mu_max = 90.0;
    static constexpr double xmin = -0.03599847736264212;
    static constexpr double xmax = 4.535808147692907;
    
    static constexpr int bias_nx = 128;
    static constexpr int bias_ny = 4;
    static constexpr int bias_nmin = 64;
    static constexpr int sigma_nx = 64;
    
    extern double *get_bsigma_coeffs();
}}  // namespace n2k::sk_globals


#endif // _N2K_SK_GLOBALS_HPP
