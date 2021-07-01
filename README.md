# Quantized Sequential Probability Ratio Tests

C implementation of the examples in 

M. Fauß, M. S. Stein, and H. V. Poor, "On Optimal Quantization in Sequential Detection."

### Build Requirements
- C Math Library \<math.h\>
- GNU Scietific Library (GSL), including development packages
- OpenBLAS, including development packages

Other CBLAS compatible BLAS libraries can be used, but the make files have to be modified accordingly. 

### Build
```
mkdir build
cd build
cmake ..
make
```

See the examples in 'main.c' for more details on how to reproduce and modify the results in the paper.

