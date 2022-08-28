#!/usr/bin/env python3

import re
import os
import sys

def uassert(cond):
    if not cond:
        print('usage: make-instantation.py dirname/kernel_NNN.cu, where NNN is e.g. 4096', file=sys.stderR)
        exit(1)

uassert(len(sys.argv) == 2)

output_filename = sys.argv[1]
m = re.match(r'kernel_(\d+)\.cu', os.path.basename(output_filename))
uassert(m is not None)

TS = int(m.group(1))
# print(f'Writing {output_filename}, TS={TS}')

with open(output_filename,'w') as fout:
    print('#include "../n2k_kernel.hpp"', file=fout)
    print('namespace {', file=fout)
    print('    #pragma nv_diag_suppress 177  // spurious nvcc warning "k declared but never referenced"', file=fout)
    print(f'    n2k::CorrelatorKernel<{TS}> k;', file=fout)
    print('}', file=fout)
