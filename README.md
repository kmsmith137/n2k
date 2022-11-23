Build instructions:
```
git clone https://github.com/kmsmith137/gputils
cd gputils
make -j
cd ..

git clone https://github.com/kmsmith137/n2k
cd n2k
make -j
./test-correlator
```

Some loose ends that I might fix later. Let me know if you'd like me to prioritize any of these.
See `n2k.hpp` for more info:

  - Currently, we only support 4+4 bit electric field samples, in the range [-7,7].
    If the value (-8) arises, then the output of the computation will be incorrect!

  - The kernel will segfault if run on a GPU which is not the cuda default device.

  - We currently use a memory layout for the output visibility matrix which is simple,
    but uses twice as much memory as necessary.

  - I may have reversed real and imaginary parts of an int4+4 (relative to the CHORD conventions)
  