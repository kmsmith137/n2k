#include "../include/n2k/pl_kernels.hpp"
#include <iostream>
#include <ksgpu/KernelTimer.hpp>

using namespace std;
using namespace ksgpu;
using namespace n2k;


static void time_pl_mask_expander(const string &name, long Tout, long Fout, long S, long ninner)
{
    const long nouter = 10;
    const long Tin = Tout/2;
    const long Fin = (Fout+3)/4;

    Array<ulong> pl_out({Tout/64,Fout,S}, af_gpu | af_zero);
    Array<ulong> pl_in({Tin/64,Fin,S}, af_gpu | af_zero);
    double gb = 8.0e-9 * ninner * (pl_out.size + pl_in.size);

    KernelTimer kt(nouter);
    
    while (kt.next()) {
        for (long i = 0; i < ninner; i++)
            launch_pl_mask_expander(pl_out, pl_in, kt.stream);  // calls CUDA_PEEK()
        if (kt.warmed_up)
            cout << name << ": Global memory BW (GB/s) = " << (gb / kt.dt) << endl;
    }
}


static void time_pl_1bit_correlator(const string &name, long T, long F, long S, long Nds, long ninner)
{
    const long nouter = 10;
    const long Tout = T / Nds;
    const long ntiles = ((S/8) * ((S/8)+1)) / 2;

    Array<ulong> pl({T/64, F, S}, af_gpu | af_zero);
    Array<uint> rfimask({F, T/32}, af_gpu | af_zero);
    Array<int> v({Tout,F,ntiles,8,8}, af_gpu | af_zero);
    double gb = 1.0e-9 * ninner * (8*pl.size + 4*v.size);
    
    KernelTimer kt(nouter);
    
    while (kt.next()) {
        for (long i = 0; i < ninner; i++)
            launch_pl_1bit_correlator(v, pl, rfimask, Nds, kt.stream);  // calls CUDA_PEEK()
        if (kt.warmed_up)
            cout << name << ": Global memory BW (GB/s) = " << (gb / kt.dt) << endl;
    }
}


int main(int argc, char **argv)
{
    // (T,F) artificially increased.
    time_pl_mask_expander("PL mask expander", 16*1024, 16*1024, 128, 10);
    time_pl_1bit_correlator("1-bit correlator (CHORD pathfinder)", 192*1024, 64*1024, 16, 128*48, 5);
    time_pl_1bit_correlator("1-bit correlator (full CHORD)", 48*1024, 32*1024, 128, 128*48, 5);
    
    return 0;
}
